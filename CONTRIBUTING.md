\# Contributing Guide



This guide explains how to set up the environment, run the code, and test the project.



---



\## Getting Started



1\. Clone the repository:

```bash

git clone <your-repo-url>

cd multi-agent-rl-mapf-drone-system

(Recommended) Create a virtual environment:



bash

Copy code

python -m venv .venv

source .venv/bin/activate   # Linux / Mac

.venv\\Scripts\\activate      # Windows

Install dependencies:



bash

Copy code

pip install -r requirements.txt

Or install in editable mode (links packages in place):



bash

Copy code

pip install -e .

Running Training

To train the PPO agent and save the model:



bash

Copy code

python main.py

This will:



Create the DroneEnv environment.



Train the PPO agent for a set number of episodes (default: 10).



Save the model weights in the models/ folder.



Print an integrity report showing drift vs hallucination error rates.



Running Tests

All tests use pytest.



Run the full test suite:



bash

Copy code

pytest -v

This will:



Run a short training + inference cycle (test\_training.py).



Verify the integrity validators (test\_integrity.py).



Use shared fixtures (conftest.py).



Example Outputs

Training output (sample):



yaml

Copy code

Starting training for 2 episodes...

Episode 1, total reward=3.40

Episode 2, total reward=4.10

Model saved to models/ppo\_drone.pt

\[Training Integrity Report] Steps=100

&nbsp; - Drift errors: 0 (0.00% of steps)

&nbsp; - Hallucination errors: 0 (0.00% of steps)

Inference output (sample):



yaml

Copy code

Step 1: action=hover, reward=1.00

Step 2: action=forward, reward=0.30

Total reward over 2 steps = 1.30

\[Inference Integrity Report] Steps=2

&nbsp; - Drift errors: 0 (0.00% of steps)

&nbsp; - Hallucination errors: 0 (0.00% of steps)

Extending the Project

Add new environment features in env/drone\_env.py.



Add new agents in agents/.



Add new tests under tests/.



Before committing changes, always run:



bash

Copy code

pytest -v

