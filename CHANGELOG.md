# Changelog

## v4.0 (2026-04-07)
- Full hardware-aware benchmarking (CPU/RAM/GPU/disk detection)
- 390+ questions across 36 categories (3-level difficulty: easy/moderate/hard)
- Multi-format support: GGUF, safetensors, PyTorch (.bin), ONNX
- Structural model classification — reads GGUF binary headers and HF config.json, no hardcoded model names
- Automatically detects and skips vision encoders, multimodal backbones, audio models, adapters, embedding models
- Fine-tunable model tracking (safetensors/bin = fine-tunable, GGUF = inference-only)
- LLM-as-Judge evaluation with sequential load/unload (no resource conflicts)
- Probabilistic scoring (0.0-1.0 continuous, not binary pass/fail)
- Advanced evaluation: code execution, self-consistency, adversarial robustness, math verification, token efficiency
- Custom domain support: `--domain "electric cars" "quantum computing"`
- Top-N model selection: `--top 2` tests only the best models
- `--fine-tunable-only` flag to test only models that support training
- GPU + CPU combined utilization (partial layer offloading)
- Automatic GPU layer calculation (full/partial/CPU-only)
- Disk space management with safe cleanup
- ChromaDB vector store for offline operation
- HuggingFace auto-download with 60-day catalog refresh
- Per-model Markdown reports with rankings
- Rich terminal output
- Curated catalog of 24 GGUF models (tiny to XL)
- Question database manager (questions.py) with ChromaDB indexing, validation, monthly update templates

## v3.0 (2026-03)
- Added hardware profiling
- Added model auto-download

## v2.0 (2026-02)
- Added multiple categories
- Added JSON/CSV export

## v1.0 (2026-01)
- Initial release — basic speed and quality testing
