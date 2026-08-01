#!/bin/bash

# List the PIDs you want to wait for
PIDS=(201403 201404)

echo "Waiting for PIDs to terminate: ${PIDS[@]}"

while true; do
    alive=()

    # Check which PIDs are still running
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive+=("$pid")
        fi
    done

    # If none left → break
    if [ ${#alive[@]} -eq 0 ]; then
        echo "All PIDs have terminated."
        break
    fi

    echo "Still running: ${alive[@]} ... checking again in 5s"
    sleep 120
done

# # ----------------------------
# # START NEXT COMMAND HERE
# # ----------------------------
# echo "Starting next job..."
# bash /data1/xw27/agent/ARLArena/examples/math_trainer/train_grpo_s_cispo.sh

bash /data/user/slime/examples/search-r1/run_qwen3_30b_a3b_seq_gspo.sh