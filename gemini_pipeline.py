"""
Gemini Emotion Classification Pipeline
======================================

A two-stage pipeline for emotion classification using the Google Gemini API.
Supports both Zero-shot and Few-shot learning strategies.

Key Components:
- DataIngestPreprocessAgent: Loads and prepares the dataset.
- TestSetClassifierAgent: Interacts with the Gemini API to classify text.
  - Manages API quota with caching and retries.
  - Compares Zero-shot vs. Few-shot performance.

Configuration:
    Requires the `GEMINI_API_KEY` environment variable to be set.

Usage:
    python gemini_pipeline.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datasets import load_dataset
import os
import time
import json
import hashlib
import random
import math
import sys
import re
import unicodedata
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Check for API Key
if "GEMINI_API_KEY" not in os.environ:
    # Check if a .env file exists and load it (optional convenience)
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if "GEMINI_API_KEY" not in os.environ:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set. "
            "Please set it or create a .env file."
        )

# ----------------- Utilities -----------------


def simple_preprocess(s: str) -> str:
    """Basic text preprocessing: lowercase, remove special chars, normalize."""
    if s is None:
        return ""
    s0 = str(s)
    s0 = unicodedata.normalize("NFKC", s0)
    s0 = s0.lower()
    s0 = re.sub(r"[^\w\s']+", " ", s0)  # Keep words and spaces
    toks = [t for t in re.split(r"\s+", s0) if t]
    toks = [t for t in toks if any(c.isalpha() for c in t)]  # Ensure some letters
    toks = [t for t in toks if len(t) > 1]  # Remove single chars
    return " ".join(toks)


# ----------------- Agent Protocol -----------------


class Agent:
    """Base class for pipeline agents."""

    name: str = "Agent"
    description: str = ""

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n=== [{self.name}] {self.description} ===\n")
        raise NotImplementedError


# ----------------- Data Agent -----------------


@dataclass
class DataIngestPreprocessAgent(Agent):
    """
    Agent to load the dataset, split it, and preprocess text.
    """

    name: str = "Data Ingest & Preprocess"
    description: str = "Load dataset, split, preprocess, store train/test DataFrames."
    dataset_name: str = "dair-ai/emotion"
    split: str = "train"
    test_size: float = 0.2
    seed: int = 42
    sample_rows: int = 5

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Loads and prepares data."""
        print(f"Loading '{self.dataset_name}'...")
        try:
            ds = load_dataset(self.dataset_name, split=self.split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return ctx

        # Split into Train/Test
        split_ds = ds.train_test_split(test_size=self.test_size, seed=self.seed)
        train = split_ds["train"]
        test = split_ds["test"]
        print(f"Train size: {len(train)}, Test size: {len(test)}")

        # Convert to Pandas
        train_df = pd.DataFrame(train)
        test_df = pd.DataFrame(test)

        # Preprocess
        train_df["processed_text"] = train_df["text"].apply(simple_preprocess)
        test_df["processed_text"] = test_df["text"].apply(simple_preprocess)

        # Describe Labels
        features = getattr(ds, "features", {})
        label_feat = features.get("label")
        label_names = None
        if label_feat and hasattr(label_feat, "names"):
            label_names = label_feat.names
            train_df["label_name"] = train_df["label"].apply(
                lambda x: label_names[int(x)]
            )
            test_df["label_name"] = test_df["label"].apply(
                lambda x: label_names[int(x)]
            )

        ctx["train_df"] = train_df
        ctx["test_df"] = test_df
        ctx["label_names"] = label_names

        print("\nOverview of processed data:")
        print(train_df[["text", "processed_text", "label_name"]].head(self.sample_rows))

        return ctx


# ----------------- Gemini Client -----------------


class _GeminiJSONClient:
    """
    Wrapper for the Google GenAI Client with JSON formatting and retry logic.
    """

    def __init__(
        self,
        model_name="gemini-1.5-flash",
        api_key_env="GEMINI_API_KEY",
        max_retries=5,
        initial_backoff=2.0,
    ):
        try:
            from google import genai

            api_key = os.getenv(api_key_env)
            if not api_key:
                raise ValueError(f"Environment variable {api_key_env} is missing.")

            self.client = genai.Client(api_key=api_key)
            print(f"Gemini client initialized (Model: {model_name})")
        except ImportError:
            raise ImportError(
                "Google GenAI library not found. Install with `pip install google-genai`."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")

        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff

    def generate_json(self, system_text: str, user_payload: Any) -> Any:
        """
        Generates content from Gemini, expecting a JSON response.
        Handles rate limits and retries.
        """
        backoff = self.initial_backoff

        for attempt in range(self.max_retries):
            try:
                # Construct Prompt
                prompt = (
                    f"{system_text}\n\n"
                    f"Input Data:\n{json.dumps(user_payload, ensure_ascii=False)}\n\n"
                    "Output specific valid JSON only."
                )

                # Call API
                response = self.client.models.generate_content(
                    model=self.model_name, contents=prompt
                )

                # Parse Response
                text = response.text
                if not text:
                    raise ValueError("Empty response from API.")

                # Strip markdown code blocks if present
                clean_text = re.sub(
                    r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE
                )

                return json.loads(clean_text)

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

                # Check for rate limit errors
                error_msg = str(e).lower()
                if "429" in error_msg or "quota" in error_msg:
                    print(f"Rate limit hit. Retrying in {backoff}s...")
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    # Non-retriable error (mostly)
                    if attempt == self.max_retries - 1:
                        print("Max retries reached.")
                        return []
                    time.sleep(1)  # Short wait for other errors

        return []


# ----------------- Classification Agent -----------------


class TestSetClassifierAgent(Agent):
    """
    Agent to classify the test set using Gemini.
    """

    name: str = "Test Set Classifier"
    description: str = "Run Zero-shot & Few-shot classification with caching."
    model_name: str = "gemini-1.5-flash"
    batch_size: int = 10
    max_test_samples: int = 50  # Limit to save quota
    seed: int = 42

    def _get_labels(self):
        return ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def _get_instructions(self, mode: str, examples: List[Dict] = None) -> str:
        """Constructs the system prompt."""
        labels = ", ".join(self._get_labels())
        base = (
            f"Classify texts into one of these emotions: {labels}.\n"
            "Return a JSON list of objects strictly following this schema:\n"
            '[{"index": int, "label": str, "confidence": float, "top_words": list[str]}]'
        )

        if mode == "few-shot" and examples:
            ex_str = json.dumps(examples, indent=2)
            return f"{base}\n\nHere are some examples:\n{ex_str}"
        return base

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        test_df = ctx.get("test_df")
        train_df = ctx.get("train_df")

        if test_df is None or train_df is None:
            print("Missing dataframes. Run Data Agent first.")
            return ctx

        # Cap test samples
        test_df = test_df.sample(
            n=min(len(test_df), self.max_test_samples), random_state=self.seed
        ).reset_index(drop=True)
        print(f"Processing {len(test_df)} test samples.")

        # Prepare Client
        try:
            client = _GeminiJSONClient(self.model_name)
        except Exception as e:
            print(f"Skipping Gemini classification: {e}")
            return ctx

        # Prepare Few-Shot Examples (one per label)
        examples = []
        for label in self._get_labels():
            sample = train_df[train_df["label_name"] == label].sample(
                1, random_state=self.seed
            )
            if not sample.empty:
                examples.append(
                    {"text": sample.iloc[0]["processed_text"], "label": label}
                )

        # Run Modes
        modes = {
            "zero-shot": self._get_instructions("zero-shot"),
            "few-shot": self._get_instructions("few-shot", examples),
        }

        results = {}

        for mode, instruction in modes.items():
            print(f"\n--- Running {mode.upper()} Classification ---")

            predictions = []
            # Batch Processing
            for i in range(0, len(test_df), self.batch_size):
                batch = test_df.iloc[i : i + self.batch_size]
                payload = [
                    {"index": idx, "text": row["processed_text"]}
                    for idx, row in batch.iterrows()
                ]

                print(f"Batch {i // self.batch_size + 1}...")
                response = client.generate_json(instruction, payload)

                if isinstance(response, list):
                    predictions.extend(response)
                else:
                    print("Invalid response format.")

            # Evaluation
            if not predictions:
                print("No predictions generated.")
                continue

            # Align predictions with ground truth
            # Creates a mapping from index to prediction
            pred_map = {p.get("index"): p for p in predictions if isinstance(p, dict)}

            y_true = []
            y_pred = []

            for idx, row in test_df.iterrows():
                y_true.append(row["label_name"])
                pred = pred_map.get(idx, {})
                y_pred.append(pred.get("label", "unknown"))

            acc = accuracy_score(y_true, y_pred)
            print(f"{mode} Accuracy: {acc:.4f}")
            results[mode] = acc

        ctx["gemini_results"] = results
        return ctx


# ----------------- Orchestrator -----------------


@dataclass
class Reporter:
    agents: List[Agent] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    def add(self, a: Agent) -> "Reporter":
        self.agents.append(a)
        return self

    def run(self) -> Dict[str, Any]:
        for a in self.agents:
            self.context = a.run(self.context)
        return self.context


# ----------------- Main -----------------


def main():
    print("Gemini Emotion Pipeline Initialized.\n")
    if "GEMINI_API_KEY" not in os.environ:
        print("Note: GEMINI_API_KEY not found. Operations relying on it will fail.")

    reporter = Reporter()
    reporter.add(DataIngestPreprocessAgent())
    reporter.add(TestSetClassifierAgent())

    reporter.run()


if __name__ == "__main__":
    main()
