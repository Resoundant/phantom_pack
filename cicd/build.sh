#!/usr/bin/env bash
set -euo pipefail

# add uv to path b/c jenkins runs as non-interactive shell
export PATH="$HOME/.local/bin:$PATH"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Remove old distributions so Jenkins only archives and uploads this build.
rm -f -- dist/*.whl dist/*.tar.gz
uv build
