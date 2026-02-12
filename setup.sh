#!/bin/bash
set -e # Exit on error

echo "🦁 Setting up Voice Cloning App..."

# Determine Python Interpreter
PYTHON_EXEC="python3.11"

# Check if we are in a Pyenv environment
if command -v pyenv >/dev/null 2>&1; then
    echo "✓ Pyenv detected."
    # If .python-version exists, pyenv respects it. If not, we might need to set it.
    
    # Check current active python version
    CURRENT_VER=$(python --version 2>&1 | cut -d ' ' -f 2)
    if [[ "$CURRENT_VER" == 3.11* ]]; then
         echo "  - Active Python is $CURRENT_VER (Good!)"
         PYTHON_EXEC="python"
    else
         echo "  - Current python is $CURRENT_VER. Checking if 3.11 is installed..."
         # Try to switch local context to 3.11
         if pyenv versions | grep -q "3.11"; then
             echo "  - Found Python 3.11 in pyenv. Setting local..."
             pyenv local 3.11.9 2>/dev/null || pyenv local 3.11
             PYTHON_EXEC="python"
         else
             echo "  - Python 3.11 not found in pyenv. Installing..."
             pyenv install 3.11.9
             pyenv local 3.11.9
             PYTHON_EXEC="python"
         fi
    fi
elif ! command -v python3.11 >/dev/null 2>&1; then
    echo "❌ Error: Python 3.11 not found and 'pyenv' is missing."
    echo "   Please install Python 3.11 or install pyenv."
    exit 1
fi

echo "✓ Using python: $($PYTHON_EXEC --version)"

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_EXEC -m venv venv
    echo "Created virtual environment 'venv'."
fi

source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (CUDA required)
echo "Installing PyTorch..."
pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install Requirements
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Local Engines
echo "Installing Engines..."
pip install -e ./F5TTS
pip install -e ./Qwen3-TTS

echo "Setup Complete! Activate context with: source venv/bin/activate"
