"""
Emotion Analysis Pipeline using Attention Rollout
=================================================

This script implements an agent-based pipeline for emotion classification using a distilbert model.
It focuses on interpretability by extracting and visualizing attention weights.

Key Components:
- PreprocessorAgent: Loads and inspects the 'emotion' dataset.
- AttentionClassificationAgent: Classifies text and calculates Attention Rollout scores.
- AttentionVisualizerAgent: Highlights words with high attention scores.
- Reporter: Orchestrates the pipeline execution.

Usage:
    python attention_analysis.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import textwrap
import sys
import os
import re
import unicodedata

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


# ---------- Pipeline Agents ----------


@dataclass
class PreprocessorAgent(Agent):
    """
    Agent responsible for loading the dataset and performing initial inspection.

    Attributes:
        dataset_name (str): Name of the dataset to load (default: "emotion").
        split (str): Dataset split to use (default: "train").
        sample_rows (int): Number of rows to sample for inspection.
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
            # Load dataset from Hugging Face
            ds: Dataset = load_dataset(self.dataset_name, split=self.split)
            ctx["dataset"] = ds
            print(
                f"Successfully loaded dataset '{self.dataset_name}' ({self.split} split)."
            )
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return ctx

        # Display Dataset Info
        print(f"\nDataset Features: {list(ds.features.keys())}")

        # Convert to DataFrame for easier inspection
        df = pd.DataFrame(ds)
        ctx["df"] = df

        print("\nSample Rows:")
        sample = df.head(self.sample_rows)
        # Retrieve label names if available
        features = getattr(ds, "features", {})
        label_feat = features.get("label")
        label_names = (
            label_feat.names if (label_feat and hasattr(label_feat, "names")) else None
        )

        for idx, row in sample.reset_index(drop=True).iterrows():
            label_val = row.get("label")
            label_str = (
                label_names[int(label_val)]
                if label_names and isinstance(label_val, (int, float))
                else str(label_val)
            )
            text_val = str(row.get("text", ""))

            print(f"\n{idx + 1}. [Label: {label_str}]")
            print(
                textwrap.fill(
                    text_val, width=100, initial_indent="    ", subsequent_indent="    "
                )
            )

        print(f"\nDataset Statistics:")
        print(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        mem_kb = df.memory_usage(deep=True).sum() / 1024
        print(f"  Memory Usage: {mem_kb:.2f} KB")

        if label_names:
            ctx["label_names"] = label_names
            print(f"\nLabel Mapping: {dict(enumerate(label_names))}")

        return ctx


# ---------- User Input Agents ----------


class UserInputAgent(Agent):
    """
    Agent to handle user input from the command line.
    Supports both direct input and piping from stdin.
    """

    task: str = "Prompt user for input sentences."
    name: str = "Input Handler"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Collects input lines from the user."""
        print(f"\n[Agent: {self.name}] {self.task}")
        lines = []

        # Check if input is being piped
        if not sys.stdin.isatty():
            data = sys.stdin.read()
            if data:
                lines = [ln for ln in data.splitlines() if ln.strip()]
        else:
            print(
                "Enter sentences to classify (one per line). Press Enter twice to finish:"
            )
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


class PreprocessingAgent(Agent):
    """
    Agent to preprocess user input text.
    Performs normalization, tokenization, stopword removal, and basic cleaning.
    """

    task: str = "Preprocess text (normalize, tokenize, remove stopwords)."
    name: str = "Text Preprocessor"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Applies NLP preprocessing steps."""
        print(f"\n[Agent: {self.name}] {self.task}")
        lines = ctx.get("user_lines", [])

        # Setup Stopwords and Lemmatizer
        stopwords = self._get_stopwords()
        lemmatize_fn = self._get_lemmatizer()

        processed_tokens_all: List[List[str]] = []
        processed_texts: List[str] = []

        for line in lines:
            if not line or not line.strip():
                continue

            # Normalization
            s = unicodedata.normalize("NFKC", line.strip())
            s = s.lower()

            # Remove punctuation (preserve apostrophes within words)
            s = re.sub(r"[^\w\s']+", " ", s)

            # Tokenization
            toks = [t for t in re.split(r"\s+", s) if t]

            # Filtering
            toks = [
                t for t in toks if re.search(r"[a-zA-Z]", t)
            ]  # Must contain letters
            toks = [t for t in toks if t not in stopwords]  # Remove stopwords

            # Lemmatization (if available)
            if lemmatize_fn:
                try:
                    toks = lemmatize_fn(toks)
                except Exception:
                    pass

            # Length check
            toks = [t for t in toks if len(t) > 1]  # Remove single chars

            if not toks:
                continue

            processed_tokens_all.append(toks)
            processed_texts.append(" ".join(toks))

        ctx["processed_tokens"] = processed_tokens_all
        ctx["processed_lines"] = processed_texts

        print("Preprocessing Complete.")
        print(f"Processed {len(processed_texts)} valid lines.")
        return ctx

    def _get_stopwords(self) -> set:
        """Retrieves stopwords from NLTK or falls back to a minimal set."""
        try:
            from nltk.corpus import stopwords as nltk_stop

            try:
                return set(nltk_stop.words("english"))
            except LookupError:
                import nltk

                nltk.download("stopwords", quiet=True)
                return set(nltk_stop.words("english"))
        except ImportError:
            # Fallback list
            return {
                "a",
                "an",
                "the",
                "and",
                "or",
                "but",
                "if",
                "while",
                "with",
                "to",
                "of",
                "in",
                "on",
                "for",
                "by",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "it",
                "this",
                "that",
                "these",
                "those",
                "i",
                "you",
                "he",
                "she",
                "they",
                "we",
                "my",
                "your",
                "his",
                "her",
                "their",
                "our",
            }

    def _get_lemmatizer(self):
        """Returns a lemmatization function (spaCy preferred, NLTK fallback)."""
        try:
            import spacy

            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
                return lambda tokens: [t.lemma_.lower() for t in nlp(" ".join(tokens))]
            except OSError:
                return None
        except ImportError:
            try:
                from nltk.stem import WordNetLemmatizer
                import nltk

                try:
                    nltk.data.find("corpora/wordnet")
                except LookupError:
                    nltk.download("wordnet", quiet=True)
                lemmatizer = WordNetLemmatizer()
                return lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]
            except Exception:
                return None


# ---------- Enhanced Classification Agent ----------


class AttentionClassificationAgent(Agent):
    """
    Agent that classifies text and extracts attention weights for interpretability.
    Uses 'distilbert-base-uncased-emotion'.
    """

    task: str = "Classify sentences and extract attention weights."
    name: str = "Emotion Classifier"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Perform classification and attention analysis."""
        print(f"\n[Agent: {self.name}] {self.task}")
        processed = ctx.get("processed_lines", [])

        if not processed:
            print("No processed text found. Skipping classification.")
            return ctx

        # Correct path for local environments if using virtualenv
        venv_site = os.path.join(
            os.path.dirname(sys.executable), "Lib", "site-packages"
        )
        if venv_site not in sys.path:
            sys.path.insert(0, venv_site)

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch.nn.functional as F
        except ImportError as e:
            print(
                f"Error: Missing required libraries ({e}). Please install 'torch' and 'transformers'."
            )
            ctx["attention_results"] = []
            return ctx

        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        print(f"Loading Model: {model_name}...")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, output_attentions=True, return_dict=True
            )
            model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
            ctx["attention_results"] = []
            return ctx

        results = []
        print("\nRunning Classification...")

        for sentence in processed:
            try:
                # Tokenize
                inputs = tokenizer(
                    sentence, return_tensors="pt", padding=True, truncation=True
                )
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                # Inference
                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = F.softmax(outputs.logits, dim=-1)
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = predictions[0][predicted_class].item()
                    attentions = outputs.attentions

                # Process Attention
                attention_data = self._compute_attention_rollout(attentions, tokens)

                emotion_label = model.config.id2label.get(
                    predicted_class, f"LABEL_{predicted_class}"
                )

                result = {
                    "sentence": sentence,
                    "original_tokens": tokens,
                    "emotion": emotion_label,
                    "confidence": confidence,
                    "attention_data": attention_data,
                    "important_words": attention_data.get("important_words", []),
                }

                results.append(result)

                # Summary for current sentence
                print(f"\nSentence: {sentence}")
                print(f"Prediction: {emotion_label} ({confidence:.2%})")
                top_words = attention_data.get("important_words", [])
                if top_words:
                    print(f"Top Attended Words: {', '.join(top_words)}")

            except Exception as e:
                print(f"Error processing sentence '{sentence[:30]}...': {e}")
                continue

        ctx["attention_results"] = results
        return ctx

    def _compute_attention_rollout(
        self, attentions: tuple, tokens: List[str]
    ) -> Dict[str, Any]:
        """
        Computes Attention Rollout to identify important words.
        Method based on Abnar & Zuidema (2020).
        """
        import numpy as _np
        import torch as _torch

        # Precaution against unexpected input types
        try:
            stacked = _torch.stack(list(attentions), dim=0)
        except Exception:
            stacked = _torch.tensor(
                _np.array(
                    [
                        (
                            a.detach().cpu().numpy()
                            if hasattr(a, "detach")
                            else _np.array(a)
                        )
                        for a in attentions
                    ]
                )
            )

        num_layers, batch, num_heads, seq_len, _ = stacked.shape

        # Average over heads -> (num_layers, batch, seq_len, seq_len)
        avg_heads = stacked.mean(dim=2)

        # Initialize rollout as identity matrix
        rollout = _torch.eye(seq_len).unsqueeze(0).repeat(batch, 1, 1)

        # Iterate through layers
        for layer in range(num_layers):
            attn = avg_heads[layer]
            # Add identity (residual connection) and normalize
            attn = attn + _torch.eye(seq_len).unsqueeze(0)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            # Matrix multiplication for rollout
            rollout = _torch.bmm(rollout, attn)

        # Attention from [CLS] token (index 0) to all other tokens
        cls_rollout = rollout[:, 0, :].cpu().numpy()[0]

        # Aggregate subword tokens into whole words
        word_scores = []
        current_word = None
        current_score = 0.0
        current_count = 0

        def _is_subword(tok: str) -> bool:
            return tok.startswith("##") or tok.startswith("▁") or tok.startswith("Ġ")

        for i, tok in enumerate(tokens):
            score = float(cls_rollout[i])
            if _is_subword(tok):
                clean = tok.lstrip("#▁Ġ")
                current_word = f"{current_word}{clean}" if current_word else clean
                current_score += score
                current_count += 1
            else:
                if current_word is not None:
                    word_scores.append(
                        (current_word, current_score / max(1, current_count))
                    )
                current_word = tok
                current_score = score
                current_count = 1

        # Determine top important words
        word_scores.sort(key=lambda x: x[1], reverse=True)
        important_words = [w for w, s in word_scores[:3]]

        return {
            "rollout_scores": cls_rollout.tolist(),
            "word_weight_pairs": word_scores,
            "important_words": important_words,
        }


# ---------- Visualization Agent ----------


class AttentionVisualizerAgent(Agent):
    """
    Agent to visualize attention weights by highlighting important words in the text.
    """

    task: str = "Visualize attention weights."
    name: str = "Visualizer"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Generates highlighted text output."""
        print(f"\n[Agent: {self.name}] {self.task}")
        results = ctx.get("attention_results", [])

        if not results:
            print("No results to visualize.")
            return ctx

        print("\n" + "=" * 60)
        print("ATTENTION ANALYSIS REPORT")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(f"\n--- Analysis {i} ---")
            print(f"Text: {result['sentence']}")
            print(
                f"Prediction: {result['emotion']} (Confidence: {result['confidence']:.2%})"
            )

            highlighted = self._highlight_text(result)
            print(f"Highlighted: {highlighted}")

            word_pairs = result.get("attention_data", {}).get("word_weight_pairs", [])
            if word_pairs:
                max_score = max((s for _, s in word_pairs), default=0.0)
                print("Top Contributing Words:")
                for rank, (word, score) in enumerate(word_pairs[:5], start=1):
                    bar_len = int((score / max_score) * 20) if max_score > 0 else 0
                    bar = "█" * bar_len
                    print(f"  {rank}. {word:<15} {score:.4f} {bar}")

        return ctx

    def _highlight_text(self, result: Dict[str, Any]) -> str:
        """Highlights important words using brackets (e.g., [WORD])."""
        sentence = result["sentence"]
        important_words = result["important_words"]

        highlighted = sentence
        # Sort by length descending to avoid replacing substrings correctly
        for word in sorted(important_words, key=len, reverse=True):
            # Simple case-insensitive replacement
            if word.lower() in sentence.lower():
                import re

                pattern = re.compile(re.escape(word), re.IGNORECASE)
                highlighted = pattern.sub(f"[{word.upper()}]", highlighted)

        return highlighted


# ---------- Orchestrator ----------


@dataclass
class Reporter:
    """Orchestrator to manage the agent pipeline."""

    agents: List[Agent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, agent: Agent) -> "Reporter":
        """Adds an agent to the pipeline."""
        self.agents.append(agent)
        return self

    def run(self) -> Dict[str, Any]:
        """Runs all agents sequentially."""
        for agent in self.agents:
            self.context = agent.run(self.context)
        return self.context


# ---------- Main Execution ----------


def main():
    print("Initializing Emotion Analysis Pipeline...\n")

    preprocess_only = "--preprocess-only" in sys.argv

    # Configure Pipeline
    reporter = Reporter()
    reporter.add(
        PreprocessorAgent(dataset_name="emotion", split="train", sample_rows=5)
    )
    reporter.add(UserInputAgent())
    reporter.add(PreprocessingAgent())

    if not preprocess_only:
        reporter.add(AttentionClassificationAgent())
        reporter.add(AttentionVisualizerAgent())

    # Execute Pipeline
    reporter.run()


if __name__ == "__main__":
    main()
