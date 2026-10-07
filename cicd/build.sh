#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Remove old distributions so Jenkins only archives and uploads this build.
rm -f -- dist/*.whl dist/*.tar.gz
uv build
