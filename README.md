# Federated Learning Sensitivity Classifier (LoRA + Flower)

This repository contains a Federated Learning system to train a binary document sensitivity classifier (Sensitive vs Non-Sensitive) using **LoRA (Low-Rank Adaptation)** and **Flower (FedAvg)**. 

It is designed to be **minimal, Windows-compatible (no Ray), and reproducible**.

## Overview
- **Model**: `distilbert-base-multilingual-cased` with LoRA adapters.
- **Federated Learning**: Flower (FedAvg protocol).
- **Privacy**: Only LoRA adapter weights (small, efficient) are exchanged; raw data never leaves the client.
- **Environment**: tested on CPU, compatible with Windows/Anaconda.

## Structure
- `src/common.py`: Model definition, LoRA setup, parameter exchange helpers.
- `src/data.py`: Data loading (supports CSV or triggers Toy dataset generation).
- `src/client.py`: Flower Client (trains local subclass of model).
- `src/server.py`: Flower Server (aggregates updates).
- `src/config.py`: Configuration (hyperparams, ports).
- `src/inference.py`: Script to test the model on new text.

## Prerequisites
- Python 3.10 or 3.11
- Anaconda or Virtualenv

## Installation

1. Create a virtual environment:
   ```bash
   conda create -n fl_env python=3.10
   conda activate fl_env
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Run the Server
Open a terminal and run:
```bash
python -m src.server
```
It will wait for clients to connect.

### 2. Run Clients
Open **two new terminals** (for two clients) and run in each:

**Client 1:**
```bash
python -m src.client --client-id 1
```

**Client 2:**
```bash
python -m src.client --client-id 2
```

The training will proceed for the configured number of rounds (default: 3).

### 3. Inference / Testing
To test the model (using the base model + initialized/trained weights logic - *note: for persistent weights after FL, saving logic needs to be extended, this repo does in-memory training demo*):

```bash
python -m src.inference --text "This is a strictly confidential document containing passwords."
```
(Note: In a real scenario, you would save the final global model on the server and load it here. By default, this script loads the base model state.)

## Using Real Data
Prepare a CSV file named `data.csv` (or any name) with columns `text` and `label` (0 or 1).
Pass it to the client:
```bash
python -m src.client --client-id 1 --data-path path/to/my_data.csv
```

## Running Tests
To verify valid installation and code correctness:
```bash
pip install -r requirements-dev.txt
pytest tests/
```

## Troubleshooting
- **Windows Path Errors**: Ensure you run commands from the repository root.
- **Port Conflicts**: Change `SERVER_ADDRESS` in `src/config.py` if 8080 is busy.
- **Transformer Warnings**: Some warnings about `weights_only=False` or "some weights not initialized" are normal for LoRA/Bert setup.

