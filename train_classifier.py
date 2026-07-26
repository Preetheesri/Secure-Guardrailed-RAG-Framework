"""
import os 
Phase 4a — Fine-tune ModernBERT as a binary "safe vs poisoned" document
classifier on the labeled splits produced in Phase 2.

Usage:
    python -m src.security.train_classifier --data data/processed/splits --out models/modernbert_security
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.utils.config import get_logger, load_config, read_jsonl, set_seed
import torch
torch.set_num_threads(os.cpu_count())
logger = get_logger(__name__)


def load_split_as_dataset(path: Path, tokenizer, max_length: int) -> Dataset:
    records = list(read_jsonl(path))
    texts = [r["context"] for r in records]
    labels = [r["label"] for r in records]
    ds = Dataset.from_dict({"text": texts, "label": labels})

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    ds = ds.map(_tokenize, batched=True, remove_columns=["text"])
    return ds


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_dir", default=None)
    parser.add_argument("--out", dest="out_dir", default=None)
    parser.add_argument("--config", default="configs/pipeline.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    data_dir = Path(args.data_dir) if args.data_dir else Path(cfg["paths"]["splits_dir"])
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["models_dir"]) / "modernbert_security"
    out_dir.mkdir(parents=True, exist_ok=True)

    sec_cfg = cfg["security_classifier"]
    max_length = cfg["tokenizer"]["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(sec_cfg["base_model"])
    model = AutoModelForSequenceClassification.from_pretrained(
        sec_cfg["base_model"], num_labels=sec_cfg["num_labels"]
    )

    train_ds = load_split_as_dataset(data_dir / "train.jsonl", tokenizer, max_length)
    val_ds = load_split_as_dataset(data_dir / "validation.jsonl", tokenizer, max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        learning_rate=sec_cfg["learning_rate"],
        per_device_train_batch_size=sec_cfg["batch_size"],
        per_device_eval_batch_size=sec_cfg["batch_size"],
        num_train_epochs=sec_cfg["epochs"],
        weight_decay=sec_cfg["weight_decay"],
        warmup_ratio=sec_cfg["warmup_ratio"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting fine-tuning of %s on %d train / %d val examples", sec_cfg["base_model"], len(train_ds), len(val_ds))
    trainer.train()

    metrics = trainer.evaluate()
    logger.info("Final validation metrics: %s", metrics)

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    logger.info("Saved fine-tuned security classifier -> %s", out_dir)


if __name__ == "__main__":
    main()
