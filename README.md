# Drone RL Environment + PPO Agent

This project demonstrates:
- A custom Gym environment for a simplified drone.
- A Proximal Policy Optimization (PPO) agent implemented in PyTorch.
- Integrity validation to catch drift (values outside expected ranges) and hallucinations (invalid actions or outputs).
- Tests to confirm training and validation run end-to-end.

---

## Project Structure

project-root/
│
├── env/
│   └── drone_env.py          # Custom Gym environment (actions, features, rewards, validation)
│
├── agents/
│   └── ppo_agent.py          # PPO agent with policy + value networks, training loop, validation
│
├── main.py                   # Orchestration script: training, saving, inference
├── integrity_validators.py   # Environment + policy validators (drift and hallucination checks)
├── integrity_stats.py        # Tracks drift vs hallucination counts, prints reports
├── models/                   # Directory for saved models (created at runtime)
│
├── tests/
│   ├── test_training.py      # Integration test: train + save + inference
│   ├── test_integrity.py     # Validator tests: clean run + bad input detection
│   └── conftest.py           # PyTest fixtures (env, agent, model path)

---

## Drone Environment

**Actions (Discrete, 5):**
- hover
- climb
- turn_left
- turn_right
- forward

**State (Continuous, 5 features normalized to [0,1]):**
1. Energy level -> remaining battery  
2. Stability -> balance/steadiness  
3. Orientation -> yaw/heading direction  
4. Altitude ratio -> altitude relative to safe range  
5. Proximity -> distance to nearest obstacle or target  

**Rewards:** Computed from a "constitution" (a list of principles, e.g. safety).  
**Validation:** Every step is checked by `IntegrityValidator` for drift and hallucinations.

---

## PPO Agent

- **Architecture:** Shared backbone -> Policy head (action probabilities) + Value head (state baseline).  
- **Training loop:** Collects rollouts, computes discounted returns, applies PPO clipped update.  
- **Validation:** `PolicyIntegrityValidator` checks:
  - Probabilities ≥ 0 and sum ≈ 1  
  - Value predictions finite  
  - Actions legal in the action space  

---

## Integrity Layer

- **integrity_validators.py:** Defines reusable environment + policy validators.  
- **integrity_stats.py:** Counts drift vs hallucination separately and reports percentages.  

---

## Tests

- **test_training.py:** Runs a small training + inference cycle, ensures saving works.  
- **test_integrity.py:** Confirms validators behave correctly:
  - No errors on normal runs
  - Errors flagged on intentionally bad inputs  
- **conftest.py:** Provides fixtures to avoid boilerplate in tests.

Run all tests:
```bash
pytest -v
