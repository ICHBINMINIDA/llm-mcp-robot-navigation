# LLM + MCP robot architectures — unified experimental version

Five architectures, one robot stack. Everything that is not the research
variable (perception, thresholds, kinematics, model settings, safety, metrics)
is shared, so a difference in the results can only come from the role given to
the LLM.

## Demo - Videos (C Architecture)

https://github.com/user-attachments/assets/4ebeb4e1-cf25-4ccb-8f77-d192ecff3080

https://github.com/user-attachments/assets/340395e1-0e51-4d80-bee9-e1d9f4f4d950



## Naming

| New name | Former name | Role of the LLM |
|---|---|---|
| **A** | C | sequences *timed* primitives (full control authority) |
| **A1** | D | sequences *continuous* primitives |
| **A2** | D-cache | same as A1 + client-side decision persistence |
| **B** | E | only names the target; one skill does the navigation |
| **C** | F | plans and repairs a sequence of three skills |

A, A1 and A2 are the primitive family; B and C are the skill family.

## Files

| File | Role |
|---|---|
| `robot_common.py` | ROS 2 node, SAM perception, motion + watchdog, visualization, server metrics |
| `llm_common.py` | Ollama call, JSON handling, target extraction, stdin, emergency stop, client metrics |
| `a_server.py` / `a_client.py` | **A** — timed primitives |
| `a1_server.py` / `a1_client.py` | **A1** — continuous primitives |
| `a1_server.py` / `a2_client.py` | **A2** — same server as A1, cached decisions |
| `b_server.py` / `b_client.py` | **B** — single skill `go_to_object` |
| `c_server.py` / `c_client.py` | **C** — `search_target`, `align_to_target`, `approach_target` |
| `camera_viewer.py` | standalone live view (raw + annotated), works with all five |
| `voice_input.py` | optional VOSK wake-word input, used by C only |
| `experiment_env.sh` / `demo_env.sh` | measurement profile / demonstration profile |
| `analyze_metrics.py` | turns the JSONL traces into the comparison table |

A1 and A2 deliberately share **one** server: the only difference between those
two experiments is the client-side decision policy.

## Running

```bash
cp *.py experiment_env.sh demo_env.sh $ROBOT_WORKDIR/
cd $ROBOT_WORKDIR
source experiment_env.sh
python3 a_client.py     # or a1_client.py / a2_client.py / b_client.py / c_client.py
```

The client starts its own server over stdio. `stop` works at any moment,
including while the robot is moving. After a session:

```bash
python3 analyze_metrics.py $METRICS_PATH --csv results.csv
```

Requirements: `mcp<2` (this code uses `mcp.server.fastmcp`, removed in mcp 2.x),
plus `rclpy`, `opencv-python`, `numpy`, `requests`, `anyio`. Voice additionally
needs `vosk` and `sounddevice`.

## Visualization and voice

**Visualization is shared, and off by default.** The annotated view (mask, box,
centroid, state banner, direction/area/bbox readout) is rendered in
`robot_common.py` and published on `VISUALIZATION_TOPIC`, so A, A1, A2, B and C
all produce the same kind of figure. It is disabled while measuring: rendering
costs ~2.6 ms per frame and raw `bgr8` at 640x480 would push ~4.6 MB/s over DDS
at 5 Hz. Two safeguards keep it out of the measurements — it is off unless
`VISUALIZATION_ENABLED=true`, and when on it is rate-limited to
`VISUALIZATION_HZ` (5 Hz) and JPEG-compressed (~16 KB instead of 900 KB per
frame, i.e. 0.08 MB/s).

`camera_viewer.py` runs as a **separate process**, not a thread inside the agent
client: the OpenCV window stays in a main thread where it is supported, the
agent keeps working over SSH without `DISPLAY`, and every architecture gets the
same viewer. It shows the annotated stream and falls back to the raw camera
whenever the annotation is stale. `VIEWER_RECORD_PATH=/tmp/run.mp4` records the
session for the demonstration video.

```bash
# terminal 1                          # terminal 2
source experiment_env.sh              source experiment_env.sh
source demo_env.sh                    source demo_env.sh
python3 c_client.py                   python3 camera_viewer.py
```

**Voice is C-only and off by default.** Speech recognition errors would
otherwise be mixed into the success rate of the navigation comparison, so
`voice_input.py` is imported by `c_client.py` alone and stays disabled unless
`VOICE_ENABLED=true` and a VOSK model is present. A missing model or missing
`vosk`/`sounddevice` degrades to keyboard input with a printed reason instead of
crashing. A spoken command during a running skill is a barge-in: the robot stops
and the new goal replaces the old one. The wake state returns to WAKE after
`VOICE_COMMAND_TIMEOUT_SEC` if nothing is said.

Note: `vosk-model-small-en-us` recognises English only. Multilingual commands
need a different VOSK model — the LLM cannot translate what the recogniser never
transcribed.

## Three independent stop layers

1. `stop_robot` over MCP sets `stop_event` — fast, in-process.
2. **The stop flag** (`ROBOT_STOP_FLAG`): the client raises a file that the
   control loops and the publisher thread poll. It works even if the server is
   busy and survives a broken stdio pipe. `lc.emergency_stop()` raises the flag
   *first*, then calls `stop_robot`.
3. The motion watchdog stops the robot if commands stop being refreshed.

Any single layer is enough to stop the robot.

**The flag is latched.** While it exists, every motion tool refuses to move
(`error: emergency_stop_latched`), skills end with `interrupted_by_stop`, and
the client ends the task with outcome `emergency_stop`. Nothing the LLM can do
clears it: only an operator starting a new task in the client, or a new session
(`experiment_env.sh`), re-arms the robot. To stop from anywhere:

```bash
touch /tmp/robot_llm_stop.flag      # or: $ROBOT_STOP_FLAG
```

## What changed compared with the first implementation

### Safety and correctness

1. **Every long-running MCP tool is `async` and offloads to a worker thread**
   (`rc.run_blocking`). In mcp 1.x a synchronous tool function runs directly
   inside the server event loop, so the whole session was blocked while the
   robot moved — `stop_robot` could never be delivered in A, B and C. The
   emergency stop is now real: start a skill, type `stop` in the same client
   (or raise the stop flag from another terminal), and `/cmd_vel` goes to zero
   in milliseconds. A *second client* is not a way to test this: each client
   starts its own server process over stdio, so it would control a separate
   ROS node.
2. **Dead-man watchdog on continuous motion** (`MAX_CONTINUOUS_MOTION_SEC`).
   If the client crashes, hangs on Ollama, or the stdio pipe dies, the node
   stops the robot by itself. `keep_moving()` lets A2 prove it is still
   supervising without changing the command.
3. **The velocity setpoint is republished at 20 Hz, zeros included**, so a lost
   stop message cannot leave the driver on its last command.
4. **The robot never drives blindly while the model is thinking**: A1 pauses
   before every LLM call, A2 pauses when the cache breaks, and B/C keep a
   fixed-rate control loop under deterministic control.

### Perception (identical in all five)

5. **One perception pipeline** in `robot_common.py`: same `SAM_MAX_SIDE` (416),
   same JPEG quality, same geometry, same debug-image policy. Previously A used
   480 and always wrote a debug image, which alone biased its timings.
6. **The largest mask is always selected.** B and C used `masks[0]`, A and A1
   used the largest — a difference in perception, not in architecture.
7. **Perception runs in its own thread** (`PerceptionWorker`) for B and C, so
   SAM latency no longer freezes the control loop.
8. **Fresh-frame rule for the primitive architectures**: after a motion, A (and
   A1 when stationary) wait for the chassis to settle and for a genuinely new
   camera frame, so the LLM never reasons about a pre-motion image. This was the
   main cause of left/right oscillation.

### LLM handling (identical in all five)

9. **`num_ctx = 4096`, `num_predict = 120`.** A2 previously ran with
   `num_ctx = 768`, smaller than its own system prompt: Ollama silently
   truncated the rules. `num_predict = 40` also truncated the JSON answer — that
   is why brace repair was needed. Repairs are still handled, but now **counted**
   (`json_repairs`) instead of hidden.
10. **Arguments are rebuilt from scratch** after validation, so an invented key
    cannot make the server reject the call, and the target cannot drift.
11. **Repair prompts keep the full context** (target + history), instead of
    asking the model to choose blindly.
12. **A deterministic fallback action** replaces the previous hard failure, so a
    single bad answer does not abort a trial (logged as `fallback_action`).
13. **One permanent stdin reader** (`Terminal`) in every client. The old
    `asyncio.to_thread(input, ...)` pattern leaked blocked threads that later
    swallowed user commands.

### Architecture-specific

* **A** — `MIN_TURN_DEG = 8` (below the motor dead-band nothing moves); the
  search rule in the prompt is a single unambiguous table.
* **A1** — one shared server with A2; `ANGULAR_SPEED = 0.30` everywhere (the old
  D server used 0.06, i.e. ~3.4 °/s, which by itself explained any apparent
  advantage of the cached variant).
* **A2** — the dead server-side detection cache is gone (it was written and
  never read). The cache that is actually studied — decision persistence — is
  bounded by a distance/rotation budget, breaks before the stop condition
  (`PRE_STOP_FACTOR`) and before the centre band (`PRE_CENTER_*`).
* **B** — perception thread + fixed-rate control; `CENTER_GO_TH = 0.15` for a
  straighter approach; a grace period stops a single missed frame from throwing
  the robot back into search.
* **C** — the important fix: `align_to_target` and `approach_target` no longer
  abort on the *first* missed detection (`last_seen` used to start at 0.0). They
  tolerate `MISS_GRACE_SEC` **and** `MAX_CONSECUTIVE_MISSES`. `MAX_STEPS` is 10
  (`C_MAX_AGENT_STEPS`, separate from the primitives' `MAX_AGENT_STEPS`) so
  recovery is possible, and the prompt states the recovery policy explicitly,
  which is what makes the planner's behaviour measurable.

## Metrics

Both sides append to the same JSONL file (`METRICS_PATH`). Per run:

`outcome`, `elapsed_sec`, `llm_calls`, `llm_seconds`, `tool_calls`,
`detect_calls`, `steps`, `json_repairs`, `json_failures`, `invalid_actions`.

A run starts when the command is typed, *before* target extraction, so the
extraction LLM call is part of every run's `llm_calls`, `llm_seconds` and
`elapsed_sec`. This matters most for B, where target extraction is the LLM's
only job.

`analyze_metrics.py` reports, per architecture: success rate, median and mean
time to target, mean LLM/tool/detection calls, the share of wall-clock time
spent waiting for the model (`llm_time_share`), and the JSON repair rate. Rows
appear under the new names A, A1, A2, B and C.

## Suggested protocol

* Same start pose, same object placements, N ≥ 10 trials per architecture,
  interleaved (A, A1, A2, B, C, then repeat) so that lighting drift affects all
  architectures equally.
* `WRITE_DEBUG_IMAGE=false` and `VISUALIZATION_ENABLED=false` while measuring.
* Keep one Ollama instance warm (`OLLAMA_KEEP_ALIVE=30m`) and discard the first
  trial after each model load.
* Record failures by category — the `outcome` field already separates
  `max_steps`, `timeout`, `target_not_visible`, `user_stop` and `error`.

## Tests

```bash
pip install "mcp<2" anyio requests numpy opencv-python-headless pytest
pytest -q tests
```

The tests run without ROS: `tests/conftest.py` replaces `rclpy` and the message
packages with small stand-ins, while the stop flag, motion state, metrics and
JSON handling run for real. They cover the latched emergency stop, run metrics,
JSON repair, C's step budget and Python 3.10 syntax for every module.
