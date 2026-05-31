# Track Plan: mem0 Integration Fine-Tuning and Retrieval

## Phase 1: mem0 Embedding Model Configuration
- [x] Task: Pin mem0 retrieval embeddings to `LiquidAI/LFM2-ColBERT-350M`.
- [x] Task: Add checked config at `training/mem0/mem0_lfm2_colbert_config.json`.
- [x] Task: Document late-interaction adapter requirement for multi-vector token embeddings and MaxSim scoring.
- [x] Task: Verification: Add tests that reject treating the model as a single-vector embedding provider.

## Phase 2: Fact Extraction Training Data Preparation
- [ ] Task: Download dialogue-to-fact datasets (such as mem0 default datasets).
- [ ] Task: Parse datasets to strictly match the LFM chat template (`<|im_start|>system`, etc.).
- [ ] Task: Verification: Verify dataset parses via Python check script.

## Phase 3: LFM Fact Extraction LoRA Training
- [ ] Task: Configure LoRA parameters (r=16, alpha=32, target modules: all linear layers).
- [ ] Task: Run training run for 3 epochs using Hugging Face Trainer.
- [ ] Task: Verification: Check training checkpoints save under `training/fine-tuning/lfm2.5-1.2b-mem0-lora`.

## Phase 4: LFM2-ColBERT Retrieval Fine-Tuning
- [x] Task: Add `training/mem0/lfm2_colbert_finetune.py` entrypoint for PyLate contrastive triplet training.
- [x] Task: Add `training/mem0/lfm2_colbert_sidecar.py` HTTP sidecar for mem0 upsert/search.
- [x] Task: Start lazy-loading sidecar on `http://127.0.0.1:8766` and verify `/health` plus `/mem0/config`.
- [ ] Task: Build or download mem0-style query/positive/negative triplets.
- [ ] Task: Run LFM2-ColBERT fine-tuning and save adapters under `training/fine-tuning/lfm2-colbert-350m-mem0`.
- [ ] Task: Verification: Evaluate MaxSim retrieval quality on held-out mem0 query/fact pairs.

## Phase 5: local mem0 Verification
- [ ] Task: Convert the adapter model to GGUF format and load it into local Ollama.
- [ ] Task: Run a memory creation curl query against `http://localhost:8765/api/v1/memories`.
- [ ] Task: Verification: Confirm that memories are extracted, records are saved to Chroma, and searches route through the LFM2-ColBERT adapter.
