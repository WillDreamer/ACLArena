#!/bin/bash
# run a command on the sdb, stripping ssh noise
ssh -p 1053 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR root@localhost "$@" 2>&1 | grep -vE '^\]0;'
