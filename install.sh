#!/usr/bin/env bash
set -e

REPO="joaocardosodias/Graphite"
INSTALL_DIR="${HOME}/.local/bin"

echo "=========================================================="
echo "  🚀 Installing Graphite (GraphRAG Embedded Engine)"
echo "=========================================================="

# 1. Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM="unknown-linux-gnu";;
    Darwin*)    PLATFORM="apple-darwin";;
    *)          echo "Error: Unsupported operating system ${OS}."; exit 1;;
esac

# 2. Detect Architecture
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64|amd64)   TARGET_ARCH="x86_64";;
    aarch64|arm64)  TARGET_ARCH="aarch64";;
    *)              echo "Error: Unsupported CPU architecture ${ARCH}."; exit 1;;
esac

TARGET="${TARGET_ARCH}-${PLATFORM}"
echo "Detected target: ${TARGET}"

# 3. Get latest release tag from GitHub API
RELEASE_TAG=$(curl -s "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")

if [ -z "${RELEASE_TAG}" ]; then
    RELEASE_TAG="v0.1.0"
fi

VERSION="${RELEASE_TAG#v}"
ARCHIVE_NAME="graphite-v${VERSION}-${TARGET}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"

echo "Fetching ${DOWNLOAD_URL}..."

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if curl -fSL -o "${TMP_DIR}/${ARCHIVE_NAME}" "${DOWNLOAD_URL}"; then
    tar -xzf "${TMP_DIR}/${ARCHIVE_NAME}" -C "${TMP_DIR}"
    
    mkdir -p "${INSTALL_DIR}"
    
    # Extract binary
    find "${TMP_DIR}" -type f -name "graphite" -exec cp {} "${INSTALL_DIR}/graphite" \;
    
    chmod +x "${INSTALL_DIR}/graphite"
    
    echo ""
    echo "✅ Successfully installed:"
    echo "   • ${INSTALL_DIR}/graphite"
    echo ""
    
    # Check PATH
    if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
        echo "⚠️  Note: Add ${INSTALL_DIR} to your PATH to run commands directly:"
        echo "   export PATH=\"${INSTALL_DIR}:\$PATH\""
        echo ""
    fi
    
    echo "Run 'graphite --help' to get started!"
else
    echo ""
    echo "❌ Failed to download pre-compiled binary for ${TARGET}."
    echo "Falling back to local Cargo install if available..."
    if command -v cargo >/dev/null 2>&1; then
        cargo install --git "https://github.com/${REPO}" graphite-cli
        echo "✅ Installed via Cargo!"
    else
        echo "Please install Rust and run: cargo install --git https://github.com/${REPO} graphite-cli"
        exit 1
    fi
fi
