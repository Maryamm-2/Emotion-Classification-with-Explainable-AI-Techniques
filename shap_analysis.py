"""
SHAP-based Emotion Interpretation
=================================

This script integrates SHAP (SHapley Additive exPlanations) to interpret the emotion classification model.
It computes feature importance values for each token, providing a game-theoretic explanation for predictions.

Key Components:
- SHAPExplainerAgent: Computes SHAP values using the Partition Explainer.
- AttentionVisualizerAgent: Adapted to visualize SHAP values alongside attention weights (if available).

Usage:
    python shap_analysis.py
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

# Suppress Warnings
warnings.filterwarnings("ignore")

# Check for SHAP
try:
    import shap
except ImportError:
    print("Error: SHAP library not found. Please install via `pip install shap`.")
    sys.exit(1)

# ---------- Agent Protocol ----------


class Agent:
    name: str = "Base Agent"
    task: str = "Base Task"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*40}\n[Agent: {self.name}] Task: {self.task}\n{'='*40}")
        raise NotImplementedError


# ---------- SHAP Agent ----------


class SHAPExplainerAgent(Agent):
    """
    Agent to explain predictions using SHAP.
    """

    name: str = "SHAP Explainer"
    task: str = "Compute SHAP values for token importance."

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        processed = ctx.get("processed_lines", [])

        if not processed:
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

        # Define Prediction Function for SHAP
        # SHAP expects a function f(strings) -> prediction scores
        def predictor(texts):
            # Tokenize
            tv = torch.tensor(
                [
                    tokenizer.encode(
                        v, padding="max_length", max_length=500, truncation=True
                    )
                    for v in texts
                ]
            )
            attention_mask = (tv != 0).type(torch.int64).to(model.device)

            with torch.no_grad():
                outputs = model(
                    tv.to(model.device), attention_mask=attention_mask.to(model.device)
                )
                probs = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()
            return probs

        # Setup SHAP Explainer
        # We use a Partition explainer with a text masker
        masker = shap.maskers.Text(tokenizer)
        class_names = list(model.config.id2label.values())

        print("Initializing SHAP Explainer...")
        explainer = shap.Explainer(predictor, masker, output_names=class_names)

        results = []

        # Analyze samples
        for idx, text in enumerate(processed):
            print(f"\nAnalyzing instance {idx+1}: '{text}'")
            try:
                shap_values = explainer([text])

                # shap_values is an Explanation object.
                # .values shape: (batch, seq_len, classes)
                # We want the values for the predicted class

                # Get prediction
                probs = predictor([text])[0]
                pred_idx = np.argmax(probs)
                pred_label = class_names[pred_idx]

                # Extract importance for the predicted class
                # shap_values[0, :, pred_idx]
                token_values = shap_values.values[0][:, pred_idx]
                tokens = shap_values.data[0]  # The tokens used by SHAP's masker

                # Pair tokens with values
                token_pairs = list(zip(tokens, token_values))

                # Sort by absolute influence
                token_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

                print(f"Prediction: {pred_label} ({probs[pred_idx]:.2%})")
                print("Top SHAP Features:")
                for tok, val in token_pairs[:5]:
                    if tok.strip():  # Skip empty tokens
                        print(f"  {tok:<15} {val:+.4f}")

                results.append(
                    {"text": text, "prediction": pred_label, "shap_values": token_pairs}
                )

            except Exception as e:
                print(f"Error analyzing sentence: {e}")

        ctx["shap_results"] = results
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
    print("Initializing SHAP Analysis Pipeline...\n")

    reporter = Reporter()
    # Add User/Preprocessing agents if needed here
    reporter.add(SHAPExplainerAgent())

    reporter.run()


if __name__ == "__main__":
    main()
