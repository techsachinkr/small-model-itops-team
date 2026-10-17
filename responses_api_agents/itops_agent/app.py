"""Gym entrypoint for the itops agent.

The agent lives in common/gym_agent.py. Gym only finds agents at responses_api_agents/<name>/ and runs
`python app.py` from there, so this file puts the repo root on sys.path and starts common.gym_agent.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.gym_agent import ItopsAgent  # noqa: E402

if __name__ == "__main__":
    ItopsAgent.run_webserver()
