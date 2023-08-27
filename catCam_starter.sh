#!/usr/bin/env bash

echo "Executing CatPreyAnalyzer"
# Tensorflow Stuff
#export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.7/dist-packages:/home/pi/tensorflow1/models/research:/home/pi/tensorflow1/models/research/slim:/home/pi/.local/lib/python3.7/site-packages
export PYTHONPATH=$PYTHONPATH:$HOME/tensorflow1/models/research:$HOME/tensorflow1/models/research/slim
cd /home/pi/CatPreyAnalyzer
while true; do
	python3 cascade.py
	sleep 5
	#exit $?
done
