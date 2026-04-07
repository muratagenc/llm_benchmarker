# LLM Benchmark Suite v4.0

**Hardware-aware local LLM benchmarking for Ubuntu Linux.**

Automatically detects your CPU, RAM, GPU (NVIDIA/AMD/Intel), and disk space — then selects, downloads, and benchmarks only the models that fit your hardware. Generates detailed per-model Markdown reports ranked best to worst.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Ubuntu%20Linux-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Hardware-Aware**: Detects CPU (model, cores, SIMD level), RAM, GPU (NVIDIA/AMD/Intel VRAM), and disk space
- **Auto-Download**: Fetches GGUF models from HuggingFace that fit your hardware — skips what won't fit
- **GPU Offload**: Automatically calculates optimal `n_gpu_layers` (full, partial, or CPU-only)
- **Disk-Space Safe**: Cleans HF cache, old reports, and oversized models when space is low — never deletes user files
- **78 Questions**: 11 categories — math, reasoning, coding, knowledge, instruction following, common sense, language, science, context retention, safety/refusal, creativity
- **ChromaDB**: Full offline operation after first run — stores results, model metadata, hardware profiles
- **Auto-Update**: Refreshes model catalog from HuggingFace every 60 days
- **Rich Reports**: Per-model Markdown with strengths, weaknesses, use-case recommendations
- **Leaderboard**: Ranked CSV, JSON, and Markdown outputs

## Quick Start

```bash
# Install dependencies
pip install llama-cpp-python rich psutil chromadb huggingface-hub requests

# Run (auto-detects hardware, downloads models, benchmarks)
python3 llm_benchmark.py

# Quick mode (1 question per category)
python3 llm_benchmark.py --quick

# Just show hardware info
python3 llm_benchmark.py --hw-info
```

### GPU Acceleration

```bash
# NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall

# AMD ROCm
CMAKE_ARGS="-DGGML_HIPBLAS=on" pip install llama-cpp-python --force-reinstall
```

## Usage

```
python3 llm_benchmark.py                    # full auto run
python3 llm_benchmark.py --quick            # 1 question per category (fast)
python3 llm_benchmark.py --no-download      # skip model auto-download
python3 llm_benchmark.py --force-update     # force HuggingFace catalog refresh
python3 llm_benchmark.py --offline          # zero network calls
python3 llm_benchmark.py --min-models 20    # auto-download threshold
python3 llm_benchmark.py --model-filter Q4  # only test models matching string
python3 llm_benchmark.py --hw-info          # print hardware profile and exit
python3 llm_benchmark.py --no-gpu           # force CPU-only inference
python3 llm_benchmark.py --categories MATH CODING  # test specific categories only
python3 llm_benchmark.py --threads 8        # override CPU thread count
python3 llm_benchmark.py --ctx 8192         # set context window size
```

## Benchmark Categories

| Category | Questions | Tests |
|----------|-----------|-------|
| **Mathematics** | 8 | Arithmetic, word problems, algebra, percentages |
| **Reasoning** | 8 | Logic, trick questions, patterns, cognitive bias |
| **Coding** | 8 | Functions, bug finding, complexity, decorators |
| **Factual Knowledge** | 8 | Chemistry, physics, history, biology, electronics |
| **Instruction Following** | 8 | Exact formats, JSON output, constrained generation |
| **Common Sense** | 8 | Physical world, tricks, spatial reasoning |
| **Language Understanding** | 8 | Grammar, fallacies, tone, analogies |
| **Science/STEM** | 8 | Physics, chemistry, biology, signal processing |
| **Context Retention** | 5 | Memory, comprehension, code tracing |
| **Safety/Refusal** | 5 | Refuses dangerous, illegal, harmful requests |
| **Creativity** | 4 | Poetry, brainstorming, fiction, analogies |

## Hardware Detection

The benchmark automatically detects:

| Component | Detection Method |
|-----------|-----------------|
| CPU | `/proc/cpuinfo` — model, cores, threads, SIMD flags (AVX/AVX2/AVX-512/FMA) |
| RAM | `/proc/meminfo` — total, available, swap |
| NVIDIA GPU | `nvidia-smi` — name, VRAM total/free, driver, compute capability |
| AMD GPU | `rocm-smi` — name, VRAM |
| Intel GPU | `/sys/class/drm` sysfs — vendor detection |
| Disk | `os.statvfs` — free space, models directory size |

### GPU Layer Calculation

```
Model fits entirely in VRAM  →  n_gpu_layers = -1 (full offload)
Model partially fits         →  n_gpu_layers = N (fractional offload)
No GPU or insufficient VRAM  →  n_gpu_layers = 0 (CPU only)
```

## Output Files

```
~/.llm_benchmark/
├── config.json              # Settings and HuggingFace catalog cache
├── chroma_db/               # ChromaDB vector store (persistent)
└── reports/
    ├── 00_LEADERBOARD.md    # Ranked summary of all models
    ├── rank01_model_name.md # Detailed report for #1 model
    ├── rank02_model_name.md # Detailed report for #2 model
    └── ...

./
├── llm_benchmark_TIMESTAMP.json          # Full results (JSON)
├── llm_benchmark_leaderboard_TIMESTAMP.csv  # Leaderboard (CSV)
└── LLM_BENCHMARK_LEADERBOARD.md         # Leaderboard (Markdown)
```

### Per-Model Report Contents

Each model report includes:
- Identity: filename, quantization, params, HuggingFace URL, downloads, likes
- Hardware: GPU layers used, CPU threads, context size, inference mode
- Performance: overall score, grade, speed (tokens/sec)
- Category breakdown: score per category with bar chart
- Strengths and weaknesses (top/bottom 3 categories)
- Use-case recommendations
- Question-by-question results with prompts, responses, and timing

## Disk Space Management

The benchmark is disk-space aware. When space is low, it cleans (in order):

1. **HuggingFace cache blobs** (re-downloadable)
2. **Old benchmark reports** (>30 days)
3. **Python `__pycache__`** directories
4. **Oversized models** that exceed hardware limits (asks permission)

It **never** deletes: models currently being tested, user files, system files.

## Curated Model Catalog

The benchmark ships with a curated list of 24 models spanning all size ranges:

| Size Class | Models | RAM Needed |
|-----------|--------|-----------|
| Tiny (<1B) | Qwen3-0.6B, SmolLM2-360M | 0.3-0.5 GB |
| Small (1-3B) | Llama-3.2-1B, Qwen3-1.7B, Llama-3.2-3B, Phi-3-mini | 0.8-2.5 GB |
| Medium (4-9B) | Qwen3-4B, Qwen2.5-7B, Mistral-7B, Llama-3.1-8B, Gemma-2-9B | 2.9-6.1 GB |
| Large (10-16B) | Mistral-Nemo-12B, Phi-4-14B, Qwen2.5-14B | 7.7-9.4 GB |
| XL (24-70B) | Qwen2.5-32B, Llama-3.3-70B | 21.5-46 GB |

Models are automatically filtered by hardware — only what fits gets downloaded and tested.

## Requirements

- **OS**: Ubuntu Linux (22.04+ recommended)
- **Python**: 3.10+
- **RAM**: 2GB minimum (for smallest models), 8GB+ recommended
- **Disk**: 5GB minimum free space
- **GPU**: Optional — NVIDIA (CUDA), AMD (ROCm), or CPU-only

### Python Packages

```
llama-cpp-python    # Required — GGUF model inference
rich                # Optional — pretty terminal output
psutil              # Optional — process memory tracking
chromadb            # Optional — persistent vector store
huggingface-hub     # Optional — model auto-download
requests            # Optional — network operations
```

## Grading Scale

| Grade | Score | Meaning |
|-------|-------|---------|
| A+ | ≥90% | Excellent — production ready |
| A | ≥80% | Very good — reliable for most tasks |
| B+ | ≥70% | Good — solid general performance |
| B | ≥60% | Decent — acceptable for simple tasks |
| C | ≥50% | Fair — limited capability |
| D | ≥40% | Poor — struggles with most tasks |
| F | <40% | Failing — not recommended |

## Contributing

Contributions welcome! Areas of interest:
- New benchmark questions (especially domain-specific)
- Support for additional model formats (ONNX, TensorRT)
- macOS/Windows support
- Multi-GPU benchmarking
- Automated regression testing between model versions

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Murat A. Genç
