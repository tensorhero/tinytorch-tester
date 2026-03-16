# Docker Build Context Setup
#
# This Dockerfile builds tinytorch-tester using the published tester-utils from GitHub
# Build from the tinytorch-tester directory:
#   cd tinytorch-tester
#   docker build -t ghcr.io/tensorhero/tinytorch-tester .

# Stage 1: Build the Go binary
FROM golang:1.24-bookworm AS builder

WORKDIR /app

# Copy go module files first for better caching
COPY go.mod go.sum ./

# Download dependencies from GitHub
RUN go mod download

# Copy the rest of the project
COPY . .

# Build the binary
RUN CGO_ENABLED=0 GOOS=linux go build \
    -o tinytorch-tester \
    -ldflags="-s -w" \
    .

# Stage 2: Runtime image with Java and Python
FROM debian:bookworm-slim

# Install runtime dependencies:
# - default-jdk-headless: Java compiler and runtime
# - python3: Python interpreter
# - ca-certificates: for HTTPS connections
RUN apt-get update && apt-get install -y \
    default-jdk-headless \
    python3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running tests
RUN useradd -m -s /bin/bash tester

# Copy the binary from builder
COPY --from=builder /app/tinytorch-tester /usr/local/bin/tinytorch-tester

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER tester

# Default command shows help
ENTRYPOINT ["tinytorch-tester"]
CMD ["--help"]
