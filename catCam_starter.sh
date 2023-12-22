#!/usr/bin/env bash

echo "Executing CatPreyAnalyzer"
# Tensorflow Stuff
while : ; do
  python3 -m balrog-prey-analyzer
  case $? in
  255)
    echo "Exiting balrog script"
    break
    ;;
  *)
    # Whatever else will be interpreted as a restart of teh script
    echo "Restarting balrog script in 5 seconds"
    sleep 5
    true
    ;;
  esac
done
