#!/bin/bash
set -e

echo "Setting up AI Health Assistant..."

# Create virtual environment
python3 -m venv venv
echo "Virtual environment created."

# Install dependencies
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
echo "Dependencies installed."

# Train and save model artifacts
venv/bin/python train_and_save.py
echo "Model training complete."

echo ""
echo "Setup complete. Run the app with:"
echo "  venv/bin/streamlit run app/app.py"
