from setuptools import setup, find_packages

setup(
    name="multi_agent_rl_drone_system",
    version="0.1.0",
    description="Multi-agent RL drone system with custom Gym environment, PPO agent, and integrity validation.",
    author="Your Name",
    packages=find_packages(exclude=("tests", "docs")),
    install_requires=[
        "gym==0.26.2",
        "torch==2.0.1",
        "numpy==1.24.3",
        "pytest==7.4.0",
    ],
    extras_require={
        "dev": [
            "matplotlib==3.7.1",
            "jupyter==1.0.0"
        ]
    },
    python_requires=">=3.9",
)
