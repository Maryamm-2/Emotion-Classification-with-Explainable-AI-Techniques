"""
LSTM Emotion Classification Model
=================================

This script implements a deep learning approach for emotion classification using a Long Short-Term Memory (LSTM) network.
It demonstrates the entire pipeline from data ingestion to model training and evaluation using TensorFlow/Keras.
Note: This file replaces the misnamed 'LogisticRegression.py'.

Key Components:
- DataIngestAgent: Loads and preprocesses the dataset.
- LSTMTrainEvalAgent: Builds, trains, and evaluates the LSTM model.

Usage:
    python lstm_model.py
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import sys
import os
import re
import pickle

# ---------- Agent Protocol ----------


class Agent:
    name: str = "Base Agent"
    task: str = "Base Task"

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n{'='*40}\n[Agent: {self.name}] Task: {self.task}\n{'='*40}")
        raise NotImplementedError


# ---------- Data Agent ----------


@dataclass
class DataIngestAgent(Agent):
    name: str = "Data Ingest"
    task: str = "Load dataset and preparing for LSTM."

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")
        try:
            from datasets import load_dataset

            ds = load_dataset("emotion")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return ctx

        train = pd.DataFrame(ds["train"])
        test = pd.DataFrame(ds["test"])
        val = pd.DataFrame(ds["validation"])

        print(f"Loaded Data: {len(train)} train, {len(val)} val, {len(test)} test.")

        ctx["train_df"] = train
        ctx["val_df"] = val
        ctx["test_df"] = test
        return ctx


# ---------- LSTM Model Agent ----------


class LSTMTrainEvalAgent(Agent):
    """
    Agent to train and evaluate an LSTM model.
    """

    name: str = "LSTM Trainer"
    task: str = "Build, train, and evaluate LSTM model."

    # Hyperparameters
    vocab_size: int = 10000
    max_len: int = 100
    embedding_dim: int = 100
    epochs: int = 5
    batch_size: int = 32

    def run(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        print(f"\n[Agent: {self.name}] {self.task}")

        train_df = ctx.get("train_df")
        test_df = ctx.get("test_df")
        val_df = ctx.get("val_df")

        if train_df is None:
            print("No training data found.")
            return ctx

        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import Tokenizer
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
            from tensorflow.keras.models import Sequential
        except ImportError:
            print("Error: TensorFlow not installed. Attempting import anyway...")
            import tensorflow as tf  # Should fail if missing

        # Tokenization
        print("Tokenizing data...")
        tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(train_df["text"])

        X_train = pad_sequences(
            tokenizer.texts_to_sequences(train_df["text"]), maxlen=self.max_len
        )
        X_val = pad_sequences(
            tokenizer.texts_to_sequences(val_df["text"]), maxlen=self.max_len
        )
        X_test = pad_sequences(
            tokenizer.texts_to_sequences(test_df["text"]), maxlen=self.max_len
        )

        y_train = pd.get_dummies(train_df["label"]).values
        y_val = pd.get_dummies(val_df["label"]).values
        y_test = pd.get_dummies(test_df["label"]).values

        num_classes = y_train.shape[1]
        print(f"Classes: {num_classes}, Input Shape: {X_train.shape}")

        # Build Model
        model = Sequential()
        model.add(
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len)
        )
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(num_classes, activation="softmax"))

        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        print("\nModel Architecture:")
        model.summary()

        # Train
        print(f"\nTraining for {self.epochs} epochs...")
        history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
        )

        # Evaluate
        print("\nEvaluating on Test Set...")
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Loss: {loss:.4f}")

        ctx["model"] = model
        ctx["history"] = history.history
        return ctx


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
    print("Initializing LSTM Emotion Classification Pipeline...\n")

    reporter = Reporter()
    reporter.add(DataIngestAgent())
    reporter.add(LSTMTrainEvalAgent())

    reporter.run()


if __name__ == "__main__":
    main()
