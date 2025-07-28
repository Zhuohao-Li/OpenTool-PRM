#!/usr/bin/env bash
set -e                                    # 任一步错就退出

# Begin retrieval with log
ts=$(date +%Y%m%d_%H%M%S)
bash retrieval_launch.sh > "retrieval_${ts}.log" 2>&1 &
RETR_PID=$!
echo "$(date) [INFO] Waiting for retrieval port ready ..."

# Waiting for port ready

PORT=8000
READY_URL="http://localhost:${PORT}/docs"
for i in {1..720}; do
    if curl -s --head "$READY_URL" | grep -q "200 OK"; then
        echo "$(date) [INFO] retrieval 已就绪"
        break
    fi
    sleep 10
done

if ! curl -s --head "$READY_URL" | grep -q "200 OK"; then
    echo "$(date) [ERROR] $READY_URL 在限定时间内仍不可达"
    kill -TERM "$RETR_PID"
    exit 1
fi

# Begin training main script
ppo_script="${1:-train_ppo.sh}"
shift
bash "$ppo_script" "$@"

# Terminate retrieval process
kill -TERM "$RETR_PID"
# kill in force
for i in {1..12}; do
    kill -0 "$RETR_PID" 2>/dev/null || break
    sleep 5
done
kill -KILL "$RETR_PID" 2>/dev/null || true

echo "$(date) All finished！"

