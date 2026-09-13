#!/usr/bin/env bash
# Single experiment configuration for architectures A, A1, A2, B and C.
# Source this file in EVERY terminal before starting any client:
#     source experiment_env.sh
# Every architecture then runs with exactly the same perception pipeline,
# thresholds, kinematics and model settings, so the only independent variable
# is the role given to the LLM.

# ---- ROS / interfaces ------------------------------------------------------
export ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-34}"
export ROBOT_WORKDIR="${ROBOT_WORKDIR:-/home/kilab/thb/ros2/ws2425/opencv_start}"
export CAMERA_TOPIC="${CAMERA_TOPIC:-camera/image}"
export CMD_VEL_TOPIC="${CMD_VEL_TOPIC:-/cmd_vel}"
export SAM_SERVICE_URL="${SAM_SERVICE_URL:-http://localhost:15000/segment}"
export MCP_ROBOT_OUTPUT_DIR="${MCP_ROBOT_OUTPUT_DIR:-/tmp/mcp_robot}"
export METRICS_PATH="${METRICS_PATH:-$MCP_ROBOT_OUTPUT_DIR/metrics.jsonl}"

# ---- LLM (identical for all five architectures) ----------------------------
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:18080/api/generate}"
export OLLAMA_MODEL="${OLLAMA_MODEL:-gemma3:27b}"
export OLLAMA_TEMPERATURE=0.0
export OLLAMA_TOP_P=0.9
export OLLAMA_NUM_CTX=4096          # must fit system prompt + history
export OLLAMA_NUM_PREDICT=120       # must fit a complete JSON action
export OLLAMA_TIMEOUT_SEC=60
export OLLAMA_KEEP_ALIVE=30m

# ---- perception (identical) ------------------------------------------------
export SAM_MAX_SIDE=416
export SAM_JPEG_QUALITY=80
export SAM_TIMEOUT_SEC=60
export SAM_EVERY_SEC=0.10
export STALE_FRAME_SEC=1.5
export MIRROR_IMAGE=false
export WRITE_DEBUG_IMAGE=false      # set to true only for demos, never while measuring

# ---- decision thresholds (identical) ---------------------------------------
export LEFT_TH=0.40
export RIGHT_TH=0.60
export STOP_AREA_RATIO=0.05
export STOP_BBOX_H_RATIO=0.55

# ---- kinematics (identical) ------------------------------------------------
export ROBOT_LINEAR_SPEED=0.10
export ROBOT_ANGULAR_SPEED=0.30
export APPROACH_SPEED=0.10
export MIN_TURN_DEG=8
export MAX_TURN_DEG=45
export MIN_FORWARD_M=0.02
export MAX_FORWARD_M=0.30
export MAX_BACKWARD_M=0.15
export CMD_PUBLISH_HZ=20
export CONTROL_HZ=30

# ---- visualization (OFF while measuring, ON for figures and the demo) ------
export VISUALIZATION_ENABLED=false
export VISUALIZATION_TOPIC=/robot_llm/visualization
export VISUALIZATION_COMPRESSED=true    # JPEG over DDS instead of raw bgr8
export VISUALIZATION_HZ=5               # independent of the SAM rate
export VISUALIZATION_JPEG_QUALITY=70
export VISUALIZATION_FRESH_SEC=1.0      # viewer falls back to the raw camera

# ---- voice (architecture C only, chapter 4; never during the comparison) ---
export VOICE_ENABLED=false
export VOSK_MODEL_PATH="${VOSK_MODEL_PATH:-$ROBOT_WORKDIR/vosk-model-small-en-us-0.15}"
export WAKE_WORD="hi robot"
export VOICE_COMMAND_TIMEOUT_SEC=8

# ---- safety ----------------------------------------------------------------
export ROBOT_STOP_FLAG=/tmp/robot_llm_stop.flag   # transport-independent stop layer
export MAX_CONTINUOUS_MOTION_SEC=3.0   # dead-man watchdog for continuous motion
export MAX_MOTION_SEC=6.0              # ceiling for a single timed motion
export SETTLE_SEC=0.25                 # chassis settling before a post-motion detection
export FRESH_FRAME_TIMEOUT_SEC=0.8

# ---- agent budgets ---------------------------------------------------------
export MAX_AGENT_STEPS=45              # A, A1, A2 (C overrides it to 10)
export HISTORY_LIMIT=5
export MAX_REPAIR_ATTEMPTS=1

rm -f "$ROBOT_STOP_FLAG"    # never start a session with a stale stop flag

echo "Experiment environment loaded (model=$OLLAMA_MODEL, metrics=$METRICS_PATH)"
