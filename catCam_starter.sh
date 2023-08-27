#!/usr/bin/env bash

echo "Executing CatPreyAnalyzer"
# Tensorflow Stuff
export CAT_PREY_ANALYZER_PATH="$(pwd)"
while true; do
	python3 cascade.py
	sleep 5
	#exit $?
done
