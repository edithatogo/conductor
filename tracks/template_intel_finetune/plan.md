# Track Plan: Intel Chip Optimization Fine-Tuning (Template)

## Phase 1: Environment Setup
- [~] Task: Install `intel-extension-for-pytorch` matching PyTorch version.
  - Blocked 2026-05-31: active Windows Python has PyTorch `2.12.0+cpu`, pip finds no compatible Windows IPEX wheel, WSL is not installed, and current Intel docs recommend native PyTorch because IPEX is EOL by end of March 2026. See `training/fine-tuning/intel-ipex-install-diagnosis.md`.
- [x] Task: Install `openvino` and `openvino-dev` libraries.
  - Completed 2026-05-31: installed `openvino==2024.6.0` and `openvino-dev==2024.6.0` into the active Python 3.11 environment. Verified `import openvino` and PyTorch import. `pip check` still reports missing `voxcpm` dependencies unrelated to OpenVINO. See `training/fine-tuning/openvino-install-verification.md`.
- [x] Task: Authenticate local Hugging Face and GitHub CLIs.
  - Completed 2026-05-31: verified `gh auth status` is authenticated as `edithatogo` and `hf auth whoami` is authenticated as `edithatogo`. Token values were not recorded. See `training/fine-tuning/cli-auth-verification.md`.
- [~] Task: Verification: Check PyTorch loads IPEX backend successfully using `python -c "import intel_extension_for_pytorch as ipex"`.
  - Blocked 2026-05-31: verification command fails with `ModuleNotFoundError: No module named 'intel_extension_for_pytorch'`, while PyTorch imports as `2.12.0+cpu`. This confirms the IPEX install blocker. See `training/fine-tuning/ipex-verification-result.md`.

## Phase 2: Fine-Tuning Pipeline
- [x] Task: Configure LoRA script with IPEX optimizations (use BF16 mixed-precision).
  - Completed 2026-05-31: added `training/fine-tuning/intel_lora_train.py` with BF16 LoRA defaults, optional IPEX optimization, and `--require-ipex` enforcement. Added config tests in `training/tests/test_intel_lora_config.py`; `python -m pytest training\tests\test_intel_lora_config.py` passed. Full training still requires `peft`, `datasets`, and `accelerate`. See `training/fine-tuning/intel-lora-script-notes.md`.
- [x] Task: Configure Hugging Face dataset loading and streaming scripts.
  - Completed 2026-05-31: added reusable streaming dataset and tokenization helpers to `training/fine-tuning/intel_lora_train.py`, including `--dataset-split`, `--max-length`, instruction/input/output formatting, and label copying. Expanded config tests; `python -m pytest training\tests\test_intel_lora_config.py` passed with 6 tests. Full streaming runtime still requires installing `datasets`. See `training/fine-tuning/hf-streaming-dataset-notes.md`.
- [x] Task: Verification: Run 5-step test epoch to confirm weights change and checkpoints save.
  - Completed 2026-05-31: installed `peft==0.19.1`, `datasets==4.8.5`, and `accelerate==1.13.0`, added `training/fine-tuning/run_lora_smoke.py`, and ran a bounded 5-step synthetic LoRA smoke test. Adapter norm changed from `0.03983466` to `2.32895434`; checkpoint saved to `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke/adapter_smoke.pt`. `python -m pytest training\tests\test_intel_lora_config.py` passed with 6 tests. `pip check` still reports unrelated `voxcpm` dependency issues and a `datasets` version mismatch. See `training/fine-tuning/lora-smoke-verification.md`.

## Phase 3: Quantization & Compression
- [x] Task: Export fine-tuned model weights to ONNX format.
  - Completed 2026-05-31: installed `onnx==1.21.0` and `onnxscript==0.7.0`, added `training/fine-tuning/export_lora_smoke_onnx.py`, exported the smoke LoRA model to `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke/model.onnx`, and validated it with `onnx.checker.check_model`. Export report shows opset 18, 1 input, 1 output, and 5 graph nodes. See `training/fine-tuning/onnx-export-verification.md`.
- [x] Task: Convert ONNX weights to OpenVINO Intermediate Representation (.xml / .bin).
  - Completed 2026-05-31: added `training/fine-tuning/convert_onnx_to_openvino_ir.py`, converted `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke/model.onnx` to OpenVINO IR at `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke/openvino-ir/model.xml` and `.bin`, and verified reload with `ov.Core().read_model`. Reload report shows 1 input, 1 output, and 15 ops. See `training/fine-tuning/openvino-ir-conversion-verification.md`.
- [x] Task: Run OpenVINO Post-Training Optimization Tool (POT) for INT8 precision.
  - Completed 2026-05-31: added `training/fine-tuning/quantize_openvino_ir_int8.py` and quantized the OpenVINO IR smoke model using NNCF calibration data. `nncf==3.1.0` was incompatible with `openvino==2024.6.0` (`openvino.Node` missing), so NNCF was pinned to `2.8.0`; `rich` and `tzdata` were restored to `hermes-agent` pins afterward. INT8 output saved to `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke/openvino-int8/model_int8.xml` and `.bin`, with 41 ops and 11 INT8-related ops. See `training/fine-tuning/int8-quantization-verification.md`.
- [x] Task: Verification: Validate converted model computes predictions on CPU with latency metrics.
  - Completed 2026-05-31: added `training/fine-tuning/validate_openvino_cpu_latency.py` and validated FP vs INT8 OpenVINO IR on CPU over 50 measured iterations after 5 warmups. INT8 predictions were finite; INT8 mean latency was `0.061286 ms`, p95 was `0.087240 ms`, and max FP-vs-INT8 absolute difference was `0.64642227`. Metrics saved to `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke/cpu_latency.json`. `python -m pytest training\tests\test_intel_lora_config.py` still passed with 6 tests. See `training/fine-tuning/cpu-latency-verification.md`.

## Phase 4: Hub Synchronization
- [~] Task: Push final ONNX/OpenVINO model schema artifacts to Hugging Face Model Hub.
  - Blocked 2026-05-31: prepared Hugging Face model-card/upload package under `training/fine-tuning/lfm2.5-1.2b-intel-lora-smoke`, targeting `edithatogo/lfm2.5-1.2b-intel-lora`, but external repo creation/upload requires explicit visibility confirmation (`public` or `private`) before publishing local files. See `training/fine-tuning/publish-readiness.md`.
- [~] Task: Push optimization code pipelines and configurations to GitHub.
  - Blocked 2026-05-31: prepared root repository commits and publish manifest targeting `edithatogo/models_lang`, but external GitHub repo creation/push requires explicit visibility confirmation (`public` or `private`) before publishing local files. See `training/fine-tuning/publish-readiness.md`.
- [ ] Task: Verification: Confirm remote download links and schemas match local checkpoints.
