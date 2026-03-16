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

| Stage | Slug           | Description                                                                |
| ----- | -------------- | -------------------------------------------------------------------------- |
| E01   | `tensor-class` | Tensor class with factory methods, operations, and dormant gradient fields |

## License

MIT
