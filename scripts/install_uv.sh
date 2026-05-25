#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  echo "uv is already installed: $(uv --version)"
  exit 0
fi

install_url="${UV_INSTALL_URL:-https://astral.sh/uv/install.sh}"
tmp_dir="$(mktemp -d)"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "Downloading uv installer from $install_url"
curl -LsSf "$install_url" -o "$tmp_dir/install_uv.sh"

echo "Running uv installer"
sh "$tmp_dir/install_uv.sh"

echo
echo "uv installation finished."
echo "If uv is not on PATH yet, restart your shell or source your shell profile."
