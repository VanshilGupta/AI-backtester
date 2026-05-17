#!/usr/bin/env bash
# One-shot WSL setup for the AI Strategy Backtester.
# Run from inside WSL:  bash /mnt/c/Users/Vanshil/backtester/setup_wsl.sh
set -euo pipefail

PROJECT="/mnt/c/Users/Vanshil/backtester"
VENV="$HOME/.venvs/backtester"   # venv on the Linux fs (fast); code stays on C:

echo ">> Installing python venv/pip (sudo may prompt for your WSL password)..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-pip >/dev/null

echo ">> Creating virtualenv at $VENV ..."
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo ">> Installing requirements ..."
pip install -q --upgrade pip
pip install -q -r "$PROJECT/requirements.txt"

echo ">> Running smoke test ..."
cd "$PROJECT"
python smoke_test.py

cat <<EOF

============================================================
Setup complete. To launch the app next time:

  source $VENV/bin/activate
  cd $PROJECT
  export ANTHROPIC_API_KEY=sk-ant-...        # or paste it in the sidebar
  streamlit run app.py

Streamlit prints a Local URL (http://localhost:8501) — open it in
your Windows browser.
============================================================
EOF
