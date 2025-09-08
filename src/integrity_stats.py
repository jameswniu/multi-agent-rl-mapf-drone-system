"""
Integrity Stats
---------------
This module tracks errors reported by integrity validators.

What is this for?
-> After each environment step or policy decision, the validators
   may detect "drift" (values out of expected range) or
   "hallucination" (invalid/unexpected outputs).
-> This class counts how often those issues happen and prints a report.

Why separate file?
-> Keeping stats separate makes the logic reusable in main.py,
   in tests, or anywhere else we need monitoring.
"""

class IntegrityStats:
    """
    Tracks drift vs hallucination counts across multiple steps.

    Attributes:
    - total_steps -> number of steps checked
    - drift_count -> how many drift errors were detected
    - hallucination_count -> how many hallucination errors were detected
    """

    def __init__(self):
        self.total_steps = 0
        self.drift_count = 0
        self.hallucination_count = 0

    def record_env(self, info):
        """
        Record errors from the environment (DroneEnv).
        - info is the dictionary returned by env.step()
        - if 'integrity_errors' exists, loop through each error dict
        - increment counters depending on error type
        """
        if "integrity_errors" in info:
            for err in info["integrity_errors"]:
                if err["type"] == "drift":
                    self.drift_count += 1
                elif err["type"] == "hallucination":
                    self.hallucination_count += 1
        self.total_steps += 1

    def record_policy(self, errors):
        """
        Record errors from the agent's policy validator.
        - errors is a list of error dicts from PolicyIntegrityValidator
        - increment counters by type
        """
        for err in errors:
            if err["type"] == "drift":
                self.drift_count += 1
            elif err["type"] == "hallucination":
                self.hallucination_count += 1
        self.total_steps += 1

    def report(self, prefix="[Integrity Report]"):
        """
        Print a summary of stats.
        - total steps checked
        - drift error count + percentage
        - hallucination error count + percentage
        """
        if self.total_steps == 0:
            print(f"{prefix} No steps recorded.")
            return

        drift_rate = 100 * self.drift_count / self.total_steps
        hall_rate = 100 * self.hallucination_count / self.total_steps

        print(f"{prefix} Steps={self.total_steps}")
        print(f"  - Drift errors: {self.drift_count} ({drift_rate:.2f}% of steps)")
        print(f"  - Hallucination errors: {self.hallucination_count} ({hall_rate:.2f}% of steps)")
