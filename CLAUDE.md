# Robot — LLM + MCP architectures

Five experimental architectures for LLM-controlled robot navigation, sharing
one perception/motion/safety stack so results only vary by the LLM's role.
Full details in [README.md](README.md) — read it before making changes here.

## Layout

- `robot_common.py` — ROS 2 node, SAM perception, motion + watchdog, metrics (server side)
- `llm_common.py` — Ollama calls, JSON handling, stdin, emergency stop, metrics (client side)
- `a_*` / `a1_*` / `a2_*` / `b_*` / `c_*` — the five architecture pairs (server + client)
- `camera_viewer.py` — standalone live view, works with all five
- `voice_input.py` — optional VOSK voice input, used by `c_client.py` only
- `analyze_metrics.py` — turns JSONL traces into a comparison table
- `experiment_env.sh` / `demo_env.sh` — env profiles for measurement vs. demo runs

## Environment

- Python 3.10, virtualenv in `.venv/` (gitignored).
- `rclpy` is not pip-installable — it comes from the ROS 2 distro
  (`source /opt/ros/<distro>/setup.bash`). Everything else is in
  `requirements.txt`.
- This code runs on the robot's ROS 2 workstation, not necessarily where it's
  edited — `cp *.py experiment_env.sh demo_env.sh $ROBOT_WORKDIR/` before
  running (see README "Running").

## Conventions

- The five architectures must stay behaviorally comparable: perception,
  kinematics, JSON handling, and safety layers live in the two `_common.py`
  files and are intentionally shared — don't fork them per-architecture unless
  the difference *is* the research variable.
- A1 and A2 share one server (`a1_server.py`) on purpose; only the client-side
  decision policy differs.
- Three independent stop layers exist (MCP `stop_robot`, the `ROBOT_STOP_FLAG`
  file, the motion watchdog) — see README "Three independent stop layers"
  before touching anything safety-related.
- Metrics/output defaults point at `/tmp` (`MCP_ROBOT_OUTPUT_DIR`,
  `METRICS_PATH`), not the repo — keep it that way.
