# LLM Benchmark Suite v4.0

**Hardware-aware local LLM benchmarking for Ubuntu Linux.**

Automatically detects your CPU, RAM, GPU (NVIDIA/AMD/Intel), and disk space — then selects, downloads, and benchmarks only the models that fit your hardware. Supports GGUF, safetensors, and PyTorch model formats. Generates detailed per-model Markdown reports ranked best to worst.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Platform](https://img.shields.io/badge/platform-Ubuntu%20Linux-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Features

- **Hardware-Aware**: Detects CPU (model, cores, SIMD level), RAM, GPU (NVIDIA/AMD/Intel VRAM), and disk space
- **Multi-Format**: Tests GGUF (llama.cpp), safetensors, PyTorch (.bin), and ONNX models
- **Structural Model Classification**: Reads GGUF binary headers and HF config.json to classify models by structure — no hardcoded model names. Automatically detects and skips vision encoders, multimodal backbones, audio models, adapters, and embedding models
- **Fine-Tunable Tracking**: Identifies which models support fine-tuning (safetensors/bin) vs inference-only (GGUF)
- **Auto-Download**: Fetches models from HuggingFace that fit your hardware — skips what won't fit
- **GPU + CPU Combined**: Uses both GPU VRAM and CPU RAM simultaneously for partial layer offloading
- **Disk-Space Safe**: Cleans HF cache, old reports, and oversized models when space is low — never deletes user files
- **390+ Questions**: 36 categories across science, engineering, coding, reasoning, and more
- **3-Level Difficulty**: Every category has easy, moderate, and hard questions
- **Probabilistic Scoring**: 0.0-1.0 continuous scores (not binary pass/fail)
- **LLM-as-Judge**: Sequential load/unload — tested model gets full hardware, then judge model evaluates open-ended responses with full hardware. Never two models competing for resources
- **Advanced Evaluation**: Code execution verification, self-consistency testing, adversarial robustness, math computation, token efficiency, difficulty profiling
- **Custom Domains**: Add any topic at runtime with `--domain "electric cars" "quantum computing"`
- **Top-N Testing**: Rank models by quality and test only the best with `--top 2`
- **ChromaDB**: Full offline operation after first run
- **Rich Reports**: Per-model Markdown with strengths, weaknesses, use-case recommendations
- **Question Database**: Separate `questions.py` with ChromaDB indexing, validation, monthly update templates

## Quick Start

```bash
# Install dependencies
pip install llama-cpp-python rich psutil chromadb huggingface-hub requests

# For safetensors/bin models (fine-tunable):
pip install transformers torch

# Run (auto-detects hardware, downloads models, benchmarks)
python3 llm_benchmark.py

# Quick mode (1 question per category, no judge)
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
python3 llm_benchmark.py                         # full auto run
python3 llm_benchmark.py --quick                 # 1 question per category (fast)
python3 llm_benchmark.py --top 2                 # only test the 2 best models
python3 llm_benchmark.py --domain "robotics"     # add custom domain at runtime
python3 llm_benchmark.py --fine-tunable-only     # only test fine-tunable models (safetensors/bin)
python3 llm_benchmark.py --no-judge              # skip LLM-as-judge (faster, deterministic only)
python3 llm_benchmark.py --no-download           # skip model auto-download
python3 llm_benchmark.py --force-update          # force HuggingFace catalog refresh
python3 llm_benchmark.py --offline               # zero network calls
python3 llm_benchmark.py --min-models 20         # auto-download threshold
python3 llm_benchmark.py --model-filter Q4       # only test models matching string
python3 llm_benchmark.py --hw-info               # print hardware profile and exit
python3 llm_benchmark.py --no-gpu                # force CPU-only inference
python3 llm_benchmark.py --categories MATH CODING  # test specific categories only
python3 llm_benchmark.py --threads 8             # override CPU thread count
python3 llm_benchmark.py --ctx 8192              # set context window size
```

## Model Classification

The benchmarker classifies models using **structural metadata** — no hardcoded model names or architecture lists. It reads the actual binary headers (GGUF) or config.json (HF) and detects:

| Signal | What It Means |
|--------|--------------|
| Tokenizer metadata | Model processes text |
| vocab_size + hidden_layers + attention_heads | Text LLM structure |
| image_size / patch_size without tokenizer | Vision encoder (skip) |
| Companion projector/mmproj file | Multimodal backbone (skip) |
| vision_config in config.json | Multimodal model (skip) |
| < 20 tensors | Adapter/projector (skip) |
| general.type = "projector" / "adapter" | Not standalone (skip) |
| Encoder-only architecture | No text generation (skip) |

This means **any new model** — even architectures that didn't exist when the code was written — will be correctly classified by its structure.

## Benchmark Categories (36)

| Category | Questions | Difficulty |
|----------|-----------|------------|
| Mathematics | 18 | 6 easy, 6 moderate, 6 hard |
| Reasoning | 18 | Logic, trick questions, patterns |
| Coding | 18 | Functions, bugs, complexity, decorators |
| Factual Knowledge | 18 | Chemistry, physics, history, biology |
| Instruction Following | 18 | Exact formats, JSON, constrained generation |
| Common Sense | 18 | Physical world, spatial reasoning |
| Language Understanding | 18 | Grammar, fallacies, tone, analogies |
| Science/STEM | 9 | Physics, chemistry, biology, signals |
| Context Retention | 9 | Memory, comprehension, code tracing |
| Safety/Refusal | 9 | Refuses dangerous/harmful requests |
| Creativity | 9 | Poetry, brainstorming, fiction |
| Electronics | 9 | Circuits, Ohm's law, semiconductors |
| PCB Design | 9 | Trace routing, DRC, layer stackup |
| JSON Structure | 9 | Valid JSON generation |
| Multi-Step Reasoning | 9 | Chain-of-thought problems |
| Role Following | 9 | Persona adherence |
| Trading/Finance | 9 | Markets, risk, indicators |
| Android Development | 9 | Android APIs, lifecycle, Kotlin |
| Agentic Behavior | 9 | Tool use, planning, self-correction |
| Embedded Systems | 9 | Microcontrollers, RTOS, protocols |
| Hallucination Detection | 9 | Detects when model makes things up |
| Context Management | 9 | Long context, multi-turn tracking |
| Physics | 9 | Mechanics, optics, quantum |
| Chemistry | 9 | Reactions, periodic table, bonds |
| Biology | 9 | Cells, genetics, ecology |
| Geography | 9 | Countries, climate, tectonics |
| History | 9 | Events, civilizations, movements |
| Advanced Math | 9 | Calculus, linear algebra, proofs |
| Thermodynamics | 9 | Laws, entropy, heat transfer |
| Metallurgy | 9 | Alloys, crystal structures |
| Extractive Metallurgy | 9 | Smelting, leaching, refining |
| Physical Metallurgy | 9 | Phase diagrams, heat treatment |
| Metalworking | 9 | CNC, welding, forming |
| Metallography | 9 | Microscopy, grain analysis |
| Psychology | 9 | Cognitive biases, behavior |
| Psychiatry | 9 | Disorders, neurotransmitters |
| **+ Custom domains** | 9 each | Via `--domain` argument |

## Scoring System

### Probabilistic Scoring (0.0 - 1.0)
- Keyword matching with fuzzy fallback
- Numeric answer verification with tolerance
- Code execution — actually runs generated Python
- Math computation — computes correct answer in Python, compares

### LLM-as-Judge (Sequential Load/Unload)
```
For each model:
  1. Load test model → full GPU + CPU
  2. Run all questions + advanced checks
  3. Unload test model → hardware freed
  4. Load judge model (best available) → full GPU + CPU
  5. Score open-ended responses (difficulty >= 2)
  6. Unload judge → hardware freed
  7. Next model
```

Score weighting: `base * 0.75 + deterministic_advanced * 0.10 + llm_judge * 0.15`

### Advanced Evaluation
- **Code Execution**: Actually runs generated Python, checks against test cases
- **Self-Consistency**: Same question 3x — measures answer agreement
- **Adversarial Robustness**: Irrelevant detail changes shouldn't change the answer
- **Math Verification**: Computes correct answer in Python, compares
- **Token Efficiency**: Penalizes overly verbose correct answers
- **Difficulty Profiling**: Finds where the model breaks (easy/moderate/hard)

## Hardware Detection

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
Model partially fits         →  n_gpu_layers = N (fractional offload, GPU + CPU combined)
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
    └── ...

./
├── llm_benchmark_TIMESTAMP.json          # Full results (JSON)
├── llm_benchmark_leaderboard_TIMESTAMP.csv  # Leaderboard (CSV)
└── LLM_BENCHMARK_LEADERBOARD.md         # Leaderboard (Markdown)
```

## Question Database Management

Questions are stored in `questions_db.json` and managed via `questions.py`:

```bash
python3 questions.py --validate     # check database consistency
python3 questions.py --stats        # show category/difficulty distribution
python3 questions.py --reindex      # rebuild ChromaDB index
python3 questions.py --search "thermodynamics"  # semantic search
python3 questions.py --template PHYSICS  # generate monthly update template
```

### Monthly Update Workflow
1. Edit `questions_db.json` (add/modify/remove questions)
2. Run `python3 questions.py --validate` to check consistency
3. Run `python3 questions.py --reindex` to rebuild the search index
4. Run `python3 questions.py --stats` to verify distribution

## Grading Scale

| Grade | Score | Meaning |
|-------|-------|---------|
| A+ | >=90% | Excellent — production ready |
| A | >=80% | Very good — reliable for most tasks |
| B+ | >=70% | Good — solid general performance |
| B | >=60% | Decent — acceptable for simple tasks |
| C | >=50% | Fair — limited capability |
| D | >=40% | Poor — struggles with most tasks |
| F | <40% | Failing — not recommended |

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
transformers        # Optional — safetensors/bin model support
torch               # Optional — needed with transformers
```

## Contributing

Contributions welcome! Areas of interest:
- New benchmark questions (any domain)
- macOS/Windows support
- Multi-GPU benchmarking
- Automated regression testing between model versions

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

Murat A. Genc
