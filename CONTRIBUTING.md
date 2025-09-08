# Contributing Guide

Thank you for considering a contribution to **multi-agent-rl-mapf-drone-system**!  
This guide explains how to set up the environment, run the code, and contribute effectively.

---

## Getting Started

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd multi-agent-rl-mapf-drone-system
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux / Mac
   .venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Or in editable mode (links packages in place, required for tests):
   ```bash
   pip install -e .
   ```

---

## Running Training

Train the PPO agent and save the model:
```bash
python src/main.py --config configs/train.yaml
```

This will:
- Create the `DroneEnv` environment  
- Train the PPO agent for a set number of episodes  
- Save the model weights in the `models/` folder  
- Print an integrity report showing drift vs hallucination error rates  

---

## Running Tests

All tests use **pytest**.

Run the full test suite:
```bash
pytest -v
```

This will:
- Run a short training + inference cycle (`test_training.py`)  
- Verify the integrity validators (`test_integrity.py`)  
- Use shared fixtures (`conftest.py`)  

---

## Example Outputs

**Training (sample):**
```text
Starting training for 2 episodes...
Episode 1, total reward=3.40
Episode 2, total reward=4.10
Model saved to models/ppo_drone.pt
[Training Integrity Report] Steps=100
  - Drift errors: 0 (0.00% of steps)
  - Hallucination errors: 0 (0.00% of steps)
```

**Inference (sample):**
```text
Step 1: action=hover, reward=1.00
Step 2: action=forward, reward=0.30
Total reward over 2 steps = 1.30
[Inference Integrity Report] Steps=2
  - Drift errors: 0 (0.00% of steps)
  - Hallucination errors: 0 (0.00% of steps)
```

---

## Extending the Project

- Add new environment features in `src/env/drone_env.py`  
- Add new agents in `src/agents/`  
- Add new tests under `tests/`  

Before committing changes, always run:
```bash
pytest -v
```

---

## Contribution Workflow

1. **Branching**  
   - Use feature branches: `feature/<short-description>`  
   - For bug fixes: `fix/<short-description>`  

2. **Commit Messages**  
   - Use clear, imperative style:  
     - ✅ `Add PPO agent logging`  
     - ❌ `Added logs`  

3. **Pull Requests**  
   - Link related issues in your PR description  
   - Ensure tests pass (`pytest -v`)  
   - Request review from a code owner (see `CODEOWNERS`)  

4. **Code Style**  
   - Follow **PEP8**  
   - Run `flake8` before submitting PRs  

---

## Support

If you run into issues, open an [Issue](../../issues) with:  
- Steps to reproduce  
- Expected vs actual behavior  
- Logs or screenshots if applicable
