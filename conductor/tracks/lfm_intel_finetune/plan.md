# Track Plan: Intel Chip Optimization Fine-Tuning (LFM-1.2B)

## Phase 1: Environment & Dependency Setup
- [x] Task: Create python virtual environment under `models_lang/training/venv`.
  - Completed 2026-05-31: created `training/venv` and bridged it to the existing hermes-agent site-packages so the environment can import the repo's active Python toolchain.
- [~] Task: Install `torch` and classify optional `intel-extension-for-pytorch` availability.
  - Partial 2026-05-31: `torch` is importable inside `training/venv` through the site-packages bridge. `intel_extension_for_pytorch` remains unavailable on this Windows host and is now treated as a legacy optional Linux-only accelerator rather than a blocking Windows requirement.
- [~] Task: Install `openvino` and `nncf` (Neural Network Compression Framework).
  - Partial 2026-05-31: `openvino==2024.6.0` and `nncf==2.8.0` are importable in `training/venv` through the site-packages bridge, but they were not installed natively into the venv with pip.
- [~] Task: Verification: Check optional `import intel_extension_for_pytorch as ipex` availability on CPU.
  - Legacy optional 2026-05-31: verification still fails because `intel_extension_for_pytorch` is not available on Windows for the active Python pairing. The Windows path proceeds with native PyTorch BF16 training and OpenVINO/NNCF export.

## Phase 2: LFM BF16 Training
- [~] Task: Download `LiquidAI/lfm2.5-1.2b-instruct` files locally.
  - Partial 2026-05-31: downloaded the public tokenizer/config metadata files with `training/fine-tuning/download_lfm_base_metadata.py` to `C:\tmp\lfm2.5-1.2b-instruct-meta-script`. Verified with `training/fine-tuning/intel_lora_train.py --preflight-only` that tokenizer/config load locally and tokenization produces `input_ids`, `attention_mask`, and `labels`. Full checkpoint weights are still pending.
- [x] Task: Configure Hugging Face token mapping via environment variables.
  - Completed 2026-05-31: added `training/fine-tuning/download_lfm_base_metadata.py`, which resolves tokens from `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, and `HF_HUB_TOKEN`, with explicit-token override support.
- [x] Task: Write PyTorch script with IPEX optimizations applying BF16 precision.
  - Completed 2026-05-31: `training/fine-tuning/intel_lora_train.py` provides native PyTorch BF16 LoRA defaults, lazy IPEX resolution, and optional legacy IPEX optimization hooks for compatible Linux runtimes.
- [ ] Task: Run 1-epoch test run to verify adapter weights compile.
- [ ] Task: Verification: Confirm adapter saves to `training/fine-tuning/lfm2.5-1.2b-intel-lora`.

## Phase 3: OpenVINO Export & Quantization
- [ ] Task: Export base model combined with LoRA weights to ONNX.
- [ ] Task: Run OpenVINO model optimizer command (`mo`) to generate `.xml` and `.bin` IR representations.
- [ ] Task: Run NNCF Post-Training Quantization script to compile INT8 precision model.
- [ ] Task: Verification: Execute local test query using OpenVINO runtime to verify latency is <1.5s per token on CPU.

## Phase 4: Push Artifacts
- [ ] Task: Git commit and push script configurations to branch `lfm-intel-optimization`.
- [ ] Task: Push final OpenVINO model artifacts to `edithatogo/lfm2.5-1.2b-intel-lora` on Hugging Face Hub.
- [ ] Task: Verification: Verify Hugging Face repository has updated files.
