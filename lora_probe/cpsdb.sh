#!/bin/bash
# scp a local file to the sdb: cpsdb.sh <local> <remote>
scp -P 1053 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$1" root@localhost:"$2" >/dev/null 2>&1 && echo "copied $1 -> $2"
