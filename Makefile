.PHONY: build test clean run install uninstall

# Build tester
build:
	go build -o tinytorch-tester .

# Install to $GOPATH/bin (global access)
install:
	go install .

# Uninstall
uninstall:
	rm -f $(shell go env GOPATH)/bin/tinytorch-tester

# Run tests
test:
	go test -v ./...

# Clean build artifacts
clean:
	rm -f tinytorch-tester
	go clean

# Run tester (requires working directory)
run:
	go run . $(ARGS)

# Install dependencies
deps:
	go mod download
	go mod tidy

# Format code
fmt:
	go fmt ./...

# Lint code
lint:
	golangci-lint run ./...

# Test all solutions (Java + Python)
test-solutions:
	./scripts/test-all-solutions.sh
