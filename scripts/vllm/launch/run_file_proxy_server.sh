#!/bin/bash

mkdir -p logs
START_TIME=`date +%Y%m%d-%H:%M:%S`
LOG_FILE=logs/fps_${START_TIME}.log

python -u scripts/vllm/launch/file_proxy_server.py > ${LOG_FILE} 2>&1 &

sleep 0.5s
tail -f $LOG_FILE
