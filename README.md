# Secure-Guardrailed-RAG-Framework
A secure Retrieval-Augmented Generation (RAG) framework that detects and removes poisoned documents using ModernBERT before passing trusted context to a Large Language Model for safe and reliable response generation.
A complete, working pipeline that detects and removes poisoned documents in a Retrieval-Augmented Generation (RAG) system before they reach the LLM.
This is the final, fully-fixed version of the project, incorporating every fix discovered while running on a real CPU-only Windows laptop:
BGE-M3 embedding fp16 auto-disabled on CPU (was causing 5+ hour runs)
BGE-M3 only computes dense embeddings (skips slow, unused sparse/colbert reps)
Flan-T5 loaded directly via AutoModelForSeq2SeqLM (works across transformers versions)
Full 4-component security layer (Pattern Detection, ModernBERT Classifier, Entropy Detection, Embedding Anomaly Detection) combined via Detective/Judge
Config defaults set to a fast, small-scale demo run (300/60/60 split, 128 token length, 4 epochs)
Quick start — full run, from zero
bash
# 0. Setup
python -m venv .venv
.venv\Scripts\activate          (Windows)   OR   source .venv/bin/activate   (Mac/Linux)
pip install -r requirements.txt

set OMP_NUM_THREADS=4           (Windows, do this every new terminal session)
set MKL_NUM_THREADS=4

# 1. Data prep (SQuAD only — fast; add nq/marco if you have time/bandwidth)
python -m src.data_prep.download_datasets --datasets squad --out data/raw
python -m src.data_prep.create_poisoned_data --in data/raw --out data/poisoned
python -m src.data_prep.label_documents --in data/poisoned --out data/labeled

# 2. Preprocessing
python -m src.preprocessing.clean_text --in data/labeled --out data/processed/cleaned
python -m src.preprocessing.tokenize_data --in data/processed/cleaned --out data/processed/tokenized
python -m src.preprocessing.split_data --in data/processed/tokenized --out data/processed/splits

# 3. Shrink to a fast demo-scale subset (skip this if you have a GPU / lots of time)
python make_subset.py

# 4. Retrieval index (uses the small subset)
python -m src.retrieval.embed_documents --in data/processed/splits_small/train.jsonl --out data/index/embeddings.npy
python -m src.retrieval.faiss_index --embeddings data/index/embeddings.npy --out data/index/faiss.index

# 5. Train the security classifier
python -m src.security.train_classifier --data data/processed/splits_small --out models/modernbert_security

# 6. Fit the entropy baseline (needed by the Detective module)
python -m src.security.fit_entropy_baseline --data data/processed/splits_small/train.jsonl --out models/entropy_baseline.json

# 7. Evaluate
python -m src.evaluation.evaluate --model models/modernbert_security --test data/processed/splits_small/test.jsonl --out outputs/report.json

# 8. Ask a real question end-to-end (downloads flan-t5-base the first time, ~1GB)
python -m src.pipeline --question "What is the capital of France?" --out outputs/answer.json
Helper scripts (optional, but useful)
make_subset.py — shrinks a full split down to 300/60/60 records for fast CPU training
show_questions.py — prints 15 real questions from your training data (good demo queries)
extract_examples.py — pulls real safe/poisoned document pairs into two readable .txt files
batch_test.py — runs many questions through the pipeline in one process (avoids reloading models each time)
No API key needed

llm.provider in configs/pipeline.yaml defaults to "local", using the free google/flan-t5-base model for answer generation. No Anthropic API key, no cost. Switch to "anthropic" only if you want to use the paid Claude API instead.

Project structure
rag-poison-defense/
├── configs/pipeline.yaml       # every path & hyperparameter, one place
├── data/                       # raw -> poisoned -> labeled -> processed -> index
├── models/                     # trained classifier + entropy baseline
├── outputs/                    # evaluation reports + pipeline run results
├── src/
│   ├── data_prep/               # Phase 1: download, poison, label
│   ├── preprocessing/           # Phase 2: clean, tokenize, split
│   ├── retrieval/                # Phase 3: BGE-M3 embeddings + FAISS
│   ├── security/                 # Phase 4: the 4-component security layer + Detective/Judge
│   ├── llm_integration/          # Phase 5: generate answer from sanitized context
│   ├── evaluation/               # Phase 6: Accuracy/Precision/Recall/F1/ASR/FPR/FNR
│   └── pipeline.py               # ties every phase together end-to-end
└── tests/                        # unit tests for poisoning logic + Judge decision logic
