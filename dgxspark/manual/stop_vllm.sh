#!/bin/bash
#
#
ps -ef | grep $(whoami | head -c 6) | grep -i vllm | grep python| awk '{print $2}' | xargs kill -9
ps -ef | grep $(whoami | head -c 6) | grep -i vllm | grep -v grep | awk '{print $2}'| xargs kill -9
