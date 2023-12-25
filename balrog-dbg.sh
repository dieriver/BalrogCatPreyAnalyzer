#!/usr/bin/env bash

echo "Executing Balrog with extra dbg info"
SNAP_ID=1
while : ; do
  python3 -v -m balrog 2>&1 | tee /data/balrog-logs/stdout.log
  case $? in
  255)
    echo "Exiting balrog script"
    mkdir -p /data/balrog-snaps/snap"$SNAP_ID"
    mv -f /data/balrog-logs/* /data/balrog-snaps/snap"$SNAP_ID"
    ((SNAP_ID++))
    break
    ;;
  *)
    # Whatever else will be interpreted as a restart of teh script
    echo "Restarting balrog script in 3 seconds"
    mkdir -p /data/balrog-snaps/snap"$SNAP_ID"
    mv -f /data/balrog-logs/* /data/balrog-snaps/snap"$SNAP_ID"
    ((SNAP_ID++))
    sleep 3
    true
    ;;
  esac
done
