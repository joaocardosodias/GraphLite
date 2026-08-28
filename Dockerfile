# ==========================================
# Stage 1: Build (Compiler Environment)
# ==========================================
FROM rust:slim-trixie AS builder

WORKDIR /app

# Install compilation, linking, and Python bindings header dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    build-essential \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace sources
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY bindings ./bindings

# Compile release binary
RUN cargo build --release --bin graphite

# ==========================================
# Stage 2: Runtime (Minimal Production Image)
# ==========================================
FROM debian:trixie-slim AS runtime

# Install TLS certificates, SSL runtime libraries, libstdc++ and curl for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged system user and group (UID/GID 10001)
RUN groupadd -g 10001 graphite && \
    useradd -u 10001 -g graphite -s /bin/sh -m graphite

# Setup persistent database data directory
WORKDIR /data
RUN chown -R graphite:graphite /data

# Copy compiled binary from builder stage
COPY --from=builder /app/target/release/graphite /usr/local/bin/graphite

# Run as non-root user for container security
USER graphite:graphite

# Expose default HTTP REST API server port
EXPOSE 8080

# Configure health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["graphite"]
CMD ["serve", "-d", "/data/graphite.graph", "--port", "8080", "--host", "0.0.0.0"]
