"""
End-to-end pipeline: given a question, retrieve top-k candidate documents
(Phase 3), run the full Data Sanitation & Alignment Layer — Pattern
Detection + ModernBERT classifier + Entropy Detection + Embedding Anomaly
Detection, combined by the Detective/Judge modules (Phase 4) — and generate
the final answer from sanitized context only (Phase 5).

Usage:
    python -m src.pipeline --config configs/pipeline.yaml --question "Who wrote Hamlet?"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.llm_integration.generate_answer import generate_answer
from src.retrieval.retrieve import Retriever
from src.security.classify_documents import SecurityClassifier
from src.security.detective_judge import run_security_layer
from src.security.embedding_anomaly import anomaly_risk_scores, fit_dbscan_anomalies
from src.utils.config import get_logger, load_config

logger = get_logger(__name__)


class RAGPoisonDefensePipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        index_dir = Path(cfg["paths"]["index_dir"])
        splits_dir = Path(cfg["paths"]["splits_dir"])
        models_dir = Path(cfg["paths"]["models_dir"])

        self.retriever = Retriever(
            index_path=str(index_dir / "faiss.index"),
            ids_path=str(index_dir / "embeddings.ids.json"),
            docs_path=str(splits_dir / "train.jsonl"),
            embedding_model=cfg["retrieval"]["embedding_model"],
            top_k=cfg["retrieval"]["top_k"],
        )
        self.classifier = SecurityClassifier(
            model_dir=str(models_dir / "modernbert_security"),
            max_length=cfg["tokenizer"]["max_length"],
            threshold=cfg["security_classifier"]["classification_threshold"],
        )

        entropy_path = models_dir / "entropy_baseline.json"
        if entropy_path.exists():
            with open(entropy_path, "r", encoding="utf-8") as f:
                self.entropy_baseline = json.load(f)
        else:
            logger.warning(
                "No entropy baseline found at %s — run "
                "`python -m src.security.fit_entropy_baseline --data <train.jsonl>` first. "
                "Falling back to default baseline.",
                entropy_path,
            )
            self.entropy_baseline = {"mean": 0.85, "std": 0.1}

    def answer(self, question: str) -> dict:
        retrieved = self.retriever.retrieve(question)

        # Embedding anomaly detection needs a batch of vectors to cluster;
        # re-embed the small retrieved batch just for this DBSCAN pass.
        embedding_anomaly_flags = None
        if len(retrieved) >= 5:  # DBSCAN needs enough points to form clusters meaningfully
            texts = [d["context"] for d in retrieved]
            vecs = self.retriever.embedder.encode(texts)
            labels = fit_dbscan_anomalies(np.asarray(vecs), eps=0.35, min_samples=2)
            embedding_anomaly_flags = anomaly_risk_scores(labels).tolist()

        kept, verdicts = run_security_layer(
            retrieved,
            classifier=self.classifier,
            entropy_baseline=self.entropy_baseline,
            embedding_anomaly_flags=embedding_anomaly_flags,
        )

        if not kept:
            return {
                "question": question,
                "answer": "I don't have enough safe context to answer this question.",
                "retrieved": retrieved,
                "verdicts": verdicts,
            }

        answer_text = generate_answer(question, kept, self.cfg)
        return {
            "question": question,
            "answer": answer_text,
            "retrieved": retrieved,
            "kept": kept,
            "verdicts": verdicts,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    pipeline = RAGPoisonDefensePipeline(cfg)
    result = pipeline.answer(args.question)

    logger.info("Q: %s", args.question)
    logger.info("A: %s", result["answer"])
    n_poisoned = sum(1 for v in result.get("verdicts", []) if v["decision"] == "poisoned")
    logger.info("Judge flagged %d/%d retrieved docs as poisoned", n_poisoned, len(result["retrieved"]))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote result -> %s", out_path)


if __name__ == "__main__":
    main()
