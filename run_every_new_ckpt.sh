#!/bin/bash
DIR="ckpts"
inotifywait -m -e create "$DIR" --format %f | while read f
do
  if [[ "$f" = *"meta"*  ]]; then
    python3 test.py
  fi
done
