"""
Emotion Classification with Generative Explanations
===================================================

This script enhances the emotion classification pipeline by adding a generative explanation step.
It uses a small Large Language Model (LLM) to generate natural language explanations for the model's predictions,
incorporating attention-based evidence.

Key Components:
- EmotionAnalyzerAgent: Classifies text and extracts attention weights.
- ClassificationExplainerLLMAgent: Generates explanations using 'google/flan-t5-small'.
- StageReporterAgent: Prints a consolidated report of the analysis stages.

Usage:
    python llm_explainability.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import sys
import os
import re
import unicodedata
import textwrap

# ---------- Agent Protocol ----------


class Agent:
    """Base class for all agents in the pipeline."""

    name: str = "Base Agent"
    task: str = "Base Task"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task.

        Args:
            ctx (Dict[str, Any]): The shared context dictionary.

        Returns:
            Dict[str, Any]: The updated context dictionary.
        """
        print(f"\n{'='*40}\n[Agent: {self.name}] Task: {self.task}\n{'='*40}")
        raise NotImplementedError("Subclasses must implement the run method.")


# ---------- Data/Feature Inspection ----------


@dataclass
class PreprocessorAgent(Agent):
    """
    Agent responsible for loading the dataset and performing initial inspection.
    """

    dataset_name: str = "emotion"
    split: str = "train"
    sample_rows: int = 5
    name: str = "Preprocessor"
    task: str = "Load dataset, inspect cache, and display features/samples."

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Loads dataset and prints summary statistics."""
        print(f"\n{'='*60}\n[Agent: {self.name}] {self.task}\n{'='*60}")
        try:
            ds: Dataset = load_dataset(self.dataset_name, split=self.split)
            ctx["dataset"] = ds
            print(f"Dataset '{self.dataset_name}' loaded.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return ctx

        # Inspection logic same as attention_analysis.py for consistency
        df = pd.DataFrame(ds)
        ctx["df"] = df

        print(f"\nDataset Statistics: {df.shape[0]} rows, {df.shape[1]} columns")

        # Label mapping
        features = getattr(ds, "features", {})
        label_feat = features.get("label")
        if label_feat and hasattr(label_feat, "names"):
            ctx["label_names"] = label_feat.names
            print(f"Labels: {label_feat.names}")

        return ctx


# ---------- User Input ----------


class UserInputAgent(Agent):
    """
    Agent to handle user input from the command line.
    """

    task: str = "Prompt user for input sentences."
    name: str = "Input Handler"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        lines = []
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                lines = [ln for ln in data.splitlines() if ln.strip()]
        else:
            print("Enter sentences (one per line). Press Enter twice to finish:")
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if not line:
                    break
                lines.append(line)
        ctx["user_lines"] = lines
        print(f"Captured {len(lines)} sentence(s).")
        return ctx


# ---------- Preprocessing ----------


class PreprocessingAgent(Agent):
    """
    Agent to preprocess user input text.
    """

    task: str = "Preprocess text (normalize, tokenize, remove stopwords)."
    name: str = "Text Preprocessor"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        lines = ctx.get("user_lines", [])

        # ... (Implementation similar to attention_analysis.py, abbreviated for brevity) ...
        # Ideally, this logic should be in a shared utility module.
        # For now, we reimplement basic preprocessing.

        processed_texts: List[str] = []
        for line in lines:
            if not line.strip():
                continue
            s = unicodedata.normalize("NFKC", line.strip()).lower()
            s = re.sub(r"[^\w\s']+", " ", s)
            processed_texts.append(s)

        ctx["processed_lines"] = processed_texts
        print(f"Processed {len(processed_texts)} lines.")
        return ctx


# ---------- Emotion Analyzer (classification + attention math) ----------


class EmotionAnalyzerAgent(Agent):
    """
    Classifies sentences and computes attention rollout scores.
    """

    task: str = "Compute predictions and attention rollout scores."
    name: str = "Emotion Analyzer"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        processed = ctx.get("processed_lines", [])
        if not processed:
            print("No text to analyze.")
            ctx["analysis_results"] = []
            return ctx

        # Helper to ensure environment is set up
        self._setup_env()

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch.nn.functional as F
        except ImportError:
            print("Error: Missing 'torch' or 'transformers'.")
            return ctx

        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        print(f"Loading Model: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, output_attentions=True, return_dict=True
        )
        model.eval()

        results = []
        for sentence in processed:
            try:
                inputs = tokenizer(
                    sentence, return_tensors="pt", padding=True, truncation=True
                )
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
                    pred_id = int(np.argmax(probs))
                    confidence = float(probs[pred_id])
                    attentions = outputs.attentions

                emotion_label = model.config.id2label.get(pred_id, f"LABEL_{pred_id}")

                # Sort all label probabilities
                label_probs = sorted(
                    [(model.config.id2label[i], float(p)) for i, p in enumerate(probs)],
                    key=lambda x: x[1],
                    reverse=True,
                )

                # Process Attention
                attention_data = self._process_attention_weights(attentions, tokens)
                important_words = attention_data["important_words"]

                res = {
                    "sentence": sentence,
                    "original_tokens": tokens,
                    "emotion": emotion_label,
                    "confidence": confidence,
                    "label_probs": label_probs,
                    "important_words": important_words,
                    "word_weight_pairs": attention_data["word_weight_pairs"],
                }
                results.append(res)

                print(
                    f"Analyzed: '{sentence[:50]}...' -> {emotion_label} ({confidence:.2%})"
                )

            except Exception as e:
                print(f"Error analyzing sentence: {e}")

        ctx["analysis_results"] = results
        return ctx

    def _setup_env(self):
        """Ensures site-packages are in path."""
        venv_site = os.path.join(
            os.path.dirname(sys.executable), "Lib", "site-packages"
        )
        if venv_site not in sys.path:
            sys.path.insert(0, venv_site)

    def _process_attention_weights(
        self, attentions: tuple, tokens: List[str]
    ) -> Dict[str, Any]:
        """Calculates Attention Rollout."""
        import torch

        # Stack layers
        stacked = torch.stack(
            list(attentions), dim=0
        )  # (layers, batch, heads, seq, seq)
        # Average heads
        avg_heads = stacked.mean(dim=2)  # (layers, batch, seq, seq)

        num_layers, batch, seq_len, _ = avg_heads.shape
        rollout = torch.eye(seq_len).unsqueeze(0).repeat(batch, 1, 1)

        for layer in range(num_layers):
            attn = avg_heads[layer] + torch.eye(seq_len).unsqueeze(0)  # Residual
            attn = attn / attn.sum(dim=-1, keepdim=True)  # Normalize
            rollout = torch.bmm(rollout, attn)

        cls_rollout = rollout[:, 0, :].cpu().numpy()[0]

        # Map to words
        word_scores = []
        current_word = None
        current_score = 0.0
        current_count = 0

        for i, tok in enumerate(tokens):
            score = float(cls_rollout[i])
            if tok.startswith("##"):
                current_word = f"{current_word}{tok[2:]}" if current_word else tok[2:]
                current_score += score
                current_count += 1
            elif tok in ["[CLS]", "[SEP]", "[PAD]"]:
                continue
            else:
                if current_word:
                    word_scores.append(
                        (current_word, current_score / max(1, current_count))
                    )
                current_word = tok
                current_score = score
                current_count = 1

        if current_word:
            word_scores.append((current_word, current_score / max(1, current_count)))

        word_scores.sort(key=lambda x: x[1], reverse=True)
        important_words = [w for w, s in word_scores[:3]]

        return {"word_weight_pairs": word_scores, "important_words": important_words}


# ---------- Emotion Classification Presentation ----------


class EmotionClassificationAgent(Agent):
    """
    Summarizes the classification results.
    """

    task: str = "Summarize classification results."
    name: str = "Classification Summary"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        results = ctx.get("analysis_results", [])

        for i, r in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Text: {r['sentence']}")
            print(f"Prediction: {r['emotion']} (Conf: {r['confidence']:.2%})")
            print("Top Probabilities:")
            for lbl, p in r["label_probs"][:3]:
                print(f"  - {lbl}: {p:.4f}")

        ctx["classification_results"] = results
        return ctx


# ---------- Classification Explainer (LLM) ----------


class ClassificationExplainerLLMAgent(Agent):
    """
    Uses a small LLM (flan-t5-small) to generate plain-English explanations.
    It combines the prediction, confidence, and attention-based evidence words into a prompt.
    """

    task: str = "Generate natural language explanations using LLM."
    name: str = "LLM Explainer"
    model_name: str = "google/flan-t5-small"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        results = ctx.get("classification_results", [])

        if not results:
            print("No results to explain.")
            return ctx

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            print(f"Loading Explainer Model: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            model.eval()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            print(f"Model loaded on {device}.")

        except Exception as e:
            print(f"Failed to load LLM: {e}. Using fallback explanations.")
            for r in results:
                r["explanation"] = self._fallback_explanation(r)
            ctx["final_results"] = results
            return ctx

        for r in results:
            explanation = self._generate_explanation(model, tokenizer, device, r)
            r["explanation"] = explanation
            print(f"Generated explanation for: '{r['sentence'][:30]}...'")

        ctx["final_results"] = results
        return ctx

    def _generate_explanation(
        self, model, tokenizer, device, result: Dict[str, Any]
    ) -> str:
        """Generates explanation using the LLM."""
        sentence = result["sentence"]
        label = result["emotion"]
        confidence = int(result["confidence"] * 100)
        evidence = ", ".join(result["important_words"])

        # Construct Prompt
        prompt = (
            f"Explain why the text '{sentence}' is classified as {label}.\n"
            f"Confidence: {confidence}%.\n"
            f"Key words: {evidence}.\n"
            f"Explanation: The text expresses {label} because"
        )

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(
                **inputs, max_new_tokens=50, num_beams=5, early_stopping=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return f"The text expresses {label} because {generated_text}"
        except Exception as e:
            print(f"Generation error: {e}")
            return self._fallback_explanation(result)

    def _fallback_explanation(self, result: Dict[str, Any]) -> str:
        """Fallback explanation if LLM generation fails."""
        return (
            f"The model predicts **{result['emotion']}** with {result['confidence']:.0%} confidence. "
            f"It primarily focused on the words: {', '.join(result['important_words'])}."
        )


# ---------- Stage Reporter ----------


class StageReporterAgent(Agent):
    """
    Prints a consolidated report of the analysis.
    """

    task: str = "Print final consolidated report."
    name: str = "Reporter"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print("\n==================== FINAL ANALYSIS REPORT ====================")
        results = ctx.get("final_results", [])

        if not results:
            print("No results to report.")
            return ctx

        for i, r in enumerate(results, 1):
            print(f"\nSample {i}:")
            print(f"  Input:       {r['sentence']}")
            print(f"  Prediction:  {r['emotion']} ({r['confidence']:.2%})")
            print(f"  Evidence:    {', '.join(r['important_words'])}")
            print(f"  Explanation: {r['explanation']}")
            print("-" * 60)

        print("\n================== END OF REPORT =================")
        return ctx


# ---------- Orchestrator ----------


@dataclass
class Reporter:
    agents: List[Agent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, agent: Agent) -> "Reporter":
        self.agents.append(agent)
        return self

    def run(self) -> Dict[str, Any]:
        for agent in self.agents:
            self.context = agent.run(self.context)
        return self.context


# ---------- Main Execution ----------


def main():
    print("Initialize Emotion Analysis & Explanation Pipeline...\n")

    reporter = Reporter()
    reporter.add(
        PreprocessorAgent(dataset_name="emotion", split="train", sample_rows=5)
    )
    reporter.add(UserInputAgent())
    reporter.add(PreprocessingAgent())
    reporter.add(EmotionAnalyzerAgent())
    reporter.add(EmotionClassificationAgent())
    reporter.add(ClassificationExplainerLLMAgent())
    reporter.add(StageReporterAgent())

    reporter.run()


if __name__ == "__main__":
    main()
