#!/bin/bash
set -e

HOST="root@213.181.123.72"
PORT="28937"
REMOTE_RUNS="/workspace/LeWAM/runs"

RECENT=$(ssh -p "$PORT" "$HOST" "ls -1 $REMOTE_RUNS | sort | tail -1")
echo "Syncing $RECENT..."

mkdir -p "$LE_WAM_ROOT/runs/recent"
rsync -avz --delete -e "ssh -p $PORT" "$HOST:$REMOTE_RUNS/$RECENT/" "$LE_WAM_ROOT/runs/recent/"
