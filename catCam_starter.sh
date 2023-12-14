#!/usr/bin/env bash

echo "Executing CatPreyAnalyzer"
# Tensorflow Stuff
while : ; do
  python3 __init__.py
  case $? in
  255)
    echo "Exiting balrog script"
    break
    ;;
  *)
    # Whatever else will be interpreted as a restart of teh script
    echo "Restarting balrog script"
    sleep 5
    true
    ;;
  esac
done
