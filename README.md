# TinyTorch Tester

Automated testing tool for the TinyTorch course.

## Option 1: Build from Source

```bash
git clone https://github.com/tensorhero/tinytorch-tester
cd tinytorch-tester
go build .
./tinytorch-tester -s tensor-class -d ~/my-solution/java
```

**Dependencies:** Go 1.24+, Java 21+, python3

## Option 2: Docker Image

**Quick Start**

```bash
cd ~/my-solution  # your solution root (contains java/ or python/)
docker pull ghcr.io/tensorhero/tinytorch-tester:latest
docker run --rm --user $(id -u):$(id -g) -v "$(pwd):/workspace" ghcr.io/tensorhero/tinytorch-tester:latest -s tensor-class -d /workspace/java
```

**Simplified script (recommended)**

Create `test.sh` in your solution root:

```bash
#!/bin/bash
LANG=${2:-java}
docker run --rm --user $(id -u):$(id -g) -v "$(pwd):/workspace" ghcr.io/tensorhero/tinytorch-tester:latest \
  -s "${1:-tensor-class}" -d "/workspace/${LANG}"
```

Usage: `chmod +x test.sh && ./test.sh tensor-class python`

**Local build (optional)**

```bash
git clone https://github.com/tensorhero/tinytorch-tester
cd tinytorch-tester
docker build -t my-tester .
# Usage: docker run --rm --user $(id -u):$(id -g) -v ~/my-solution:/workspace my-tester -s tensor-class -d /workspace/java
```

## Stages

| Stage | Slug                        | Name                    |
| ----- | --------------------------- | ----------------------- |
| E01   | `tensor-class`              | Tensor Class            |
| E02   | `computation-graph`         | Computation Graph       |
| E03   | `backpropagation`           | Backpropagation         |
| E04   | `more-backward-ops`         | More Backward Ops       |
| E05   | `activations`               | Activations             |
| E06   | `linear-layer`              | Linear Layer            |
| E07   | `loss-functions`            | Loss Functions          |
| E08   | `optimizers`                | Optimizers              |
| E09   | `training-loop`             | Training Loop           |
| E10   | `dataloader-and-mlp`        | DataLoader & MLP        |
| E11   | `tokenization`              | Tokenization            |
| E12   | `embeddings`                | Embeddings              |
| E13   | `attention`                 | Attention               |
| E14   | `transformer-block`         | Transformer Block       |
| E15   | `gpt-and-generate`          | GPT & Generate          |
| E16   | `quantization-and-kv-cache` | Quantization & KV Cache |
| E17   | `profiling-and-compression` | Profiling & Compression |

## License

MIT
