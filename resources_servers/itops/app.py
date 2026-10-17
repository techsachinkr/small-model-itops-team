"""Gym entrypoint for the itops resources server.

The server lives in gym_env/ (CLAUDE.md "Repo layout"). Gym only finds servers at
resources_servers/<name>/ and runs `python app.py` from there, so this file puts the repo root on
sys.path and starts gym_env.server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gym_env.server import ItopsResourcesServer  # noqa: E402

if __name__ == "__main__":
    ItopsResourcesServer.run_webserver()
