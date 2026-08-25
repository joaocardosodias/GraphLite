#!/usr/bin/env bash
set -euo pipefail

REPO="joaocardosodias/Graphite"
INSTALL_DIR="${GRAPHITE_INSTALL_DIR:-${HOME}/.local/bin}"

info() {
    printf "info: %s\n" "$*"
}

error() {
    printf "error: %s\n" "$*" >&2
    exit 1
}

# Detect Operating System
OS="$(uname -s)"
case "${OS}" in
    Linux*)     PLATFORM="unknown-linux-gnu";;
    Darwin*)    PLATFORM="apple-darwin";;
    *)          error "Unsupported operating system: ${OS}";;
esac

# Detect Architecture
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64|amd64)   TARGET_ARCH="x86_64";;
    aarch64|arm64)  TARGET_ARCH="aarch64";;
    *)              error "Unsupported CPU architecture: ${ARCH}";;
esac

TARGET="${TARGET_ARCH}-${PLATFORM}"

# Resolve latest release tag
RELEASE_TAG=$(curl -s "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/' || echo "")
if [ -z "${RELEASE_TAG}" ]; then
    RELEASE_TAG="v0.1.0"
fi

VERSION="${RELEASE_TAG#v}"
ARCHIVE_NAME="graphite-v${VERSION}-${TARGET}.tar.gz"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${ARCHIVE_NAME}"

info "Downloading graphite ${RELEASE_TAG} for ${TARGET}..."

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if curl -fSL -o "${TMP_DIR}/${ARCHIVE_NAME}" "${DOWNLOAD_URL}" 2>/dev/null; then
    tar -xzf "${TMP_DIR}/${ARCHIVE_NAME}" -C "${TMP_DIR}"
    
    mkdir -p "${INSTALL_DIR}"
    find "${TMP_DIR}" -type f -name "graphite" -exec cp {} "${INSTALL_DIR}/graphite" \;
    chmod +x "${INSTALL_DIR}/graphite"
    
    info "Graphite binary installed to ${INSTALL_DIR}/graphite"
    
    case ":${PATH}:" in
        *":${INSTALL_DIR}:"*) ;;
        *) info "Add ${INSTALL_DIR} to your PATH to run 'graphite' directly." ;;
    esac
else
    info "Binary release not found for ${TARGET}. Attempting build via Cargo..."
    if command -v cargo >/dev/null 2>&1; then
        cargo install --git "https://github.com/${REPO}" graphite-cli
        info "Installed graphite via Cargo."
    else
        error "Failed to download pre-compiled binary and Cargo is not available."
    fi
fi
