#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Environnement absent. Exécutez d'abord : ./setup.sh" >&2
    exit 1
fi

echo "Interface : http://127.0.0.1:5000"
exec ./.venv/bin/python app.py
