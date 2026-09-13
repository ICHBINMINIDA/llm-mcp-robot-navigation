#!/usr/bin/env bash
# Demonstration / figures profile. Source it AFTER experiment_env.sh:
#     source experiment_env.sh && source demo_env.sh
# Never source this while collecting measurements: rendering, publishing and
# speech recognition all add latency and variance to the numbers.
export VISUALIZATION_ENABLED=true
export WRITE_DEBUG_IMAGE=true
export VOICE_ENABLED=true                 # architecture C only
echo "Demo profile: visualization ON, voice ON (do not use for measurements)"
