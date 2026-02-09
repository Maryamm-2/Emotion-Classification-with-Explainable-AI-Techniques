"""
LIME-based Emotion Explanation
==============================

This script applies Local Interpretable Model-agnostic Explanations (LIME) to the emotion classification model.
It generates post-hoc explanations by perturbing the input text and observing changes in model predictions,
thereby identifying influential words for a specific prediction.

Key Components:
- ReasoningAgent: Wraps the LIME TextExplainer to explain model predictions.
- Integration: Works with the same `distilbert-base-uncased-emotion` model used in other scripts.

Usage:
    python lime_analysis.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
from datasets import load_dataset
import sys
import os
import re
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Check for LIME
try:
    from lime.lime_text import LimeTextExplainer
except ImportError:
    print("Error: LIME library not found. Please install via `pip install lime`.")
    sys.exit(1)

# ---------- Agent Protocol ----------


class Agent:
    name: str = "Base Agent"
    task: str = "Base Task"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*40}\n[Agent: {self.name}] Task: {self.task}\n{'='*40}")
        raise NotImplementedError


# ---------- LIME Reasoning Agent ----------


class ReasoningAgent(Agent):
    """
    Agent to explain predictions using LIME.
    """

    name: str = "LIME Explainer"
    task: str = "Generate local explanations for model predictions."
    num_samples: int = 500  # Number of perturbations
    num_features: int = 10  # Max words to highlight

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Runs LIME on processed text."""
        print(f"\n[Agent: {self.name}] {self.task}")
        processed = ctx.get("processed_lines", [])

        if not processed:
            # Fallback if no prior processing
            print("No processed text found. Using sample sentences.")
            processed = [
                "i feel so happy and excited about the news",
                "this is a terrible situation and i am angry",
                "i am feeling a bit lonely and sad today",
            ]

        self._setup_env()

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import torch.nn.functional as F
        except ImportError:
            print("Missing torch/transformers.")
            return ctx

        # Load Model
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        print(f"Loading Model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()

        # Prediction Function for LIME
        def predictor(texts):
            outputs = model(
                **tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
            )
            probs = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            return probs

        # Class names
        class_names = list(model.config.id2label.values())
        explainer = LimeTextExplainer(class_names=class_names)

        results = []

        for idx, text in enumerate(processed):
            print(f"\nExplaining instance {idx+1}: '{text}'")

            try:
                exp = explainer.explain_instance(
                    text,
                    predictor,
                    num_features=self.num_features,
                    num_samples=self.num_samples,
                )

                # Get the predicted class
                probs = predictor([text])[0]
                pred_idx = np.argmax(probs)
                pred_label = class_names[pred_idx]
                confidence = probs[pred_idx]

                print(f"Prediction: {pred_label} ({confidence:.2%})")

                # Extract explanation list
                explanation_list = exp.as_list(label=pred_idx)

                # Visualize
                print("Top Influential Words:")
                for word, weight in explanation_list:
                    sign = "+" if weight > 0 else "-"
                    print(f"  {word:<15} {weight:+.4f} ({sign})")

                results.append(
                    {
                        "text": text,
                        "prediction": pred_label,
                        "lime_weights": explanation_list,
                    }
                )

            except Exception as e:
                print(f"Error explaining instance: {e}")

        ctx["lime_results"] = results
        return ctx

    def _setup_env(self):
        venv_site = os.path.join(
            os.path.dirname(sys.executable), "Lib", "site-packages"
        )
        if venv_site not in sys.path:
            sys.path.insert(0, venv_site)


# ---------- Orchestrator ----------


@dataclass
class Reporter:
    agents: List[Agent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, a: Agent) -> "Reporter":
        self.agents.append(a)
        return self

    def run(self):
        for a in self.agents:
            self.context = a.run(self.context)


# ---------- Main ----------


def main():
    print("Initializing LIME Explanation Pipeline...\n")

    # We can reuse agents from other files if we structured this as a package,
    # but for standalone execution, we'll keep it simple.

    reporter = Reporter()
    # Add User/Preprocessing agents if needed here
    reporter.add(ReasoningAgent())

    reporter.run()


if __name__ == "__main__":
    main()
