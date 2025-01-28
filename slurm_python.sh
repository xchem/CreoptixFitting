#!/bin/bash

ARGS=$@

# splashscreen
echo "************************************************************************"
echo "script         = $0"
echo "whoami         = $(whoami)"
echo "hostname       = $(hostname)"
echo "ip-address     = $(ifconfig | grep -A1 eth0 | grep inet | awk '{print $2}')"
echo "arguments      = $ARGS"
echo "************************************************************************"

# setup conda
echo 'Activating conda...'
source /dls/science/groups/i04-1/software/max/load_py310.sh

echo 'conda info...'
conda info

echo 'python location...'
which python

echo 'running python...'
python $ARGS

exit $?
