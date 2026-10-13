#!/bin/zsh
# Restart everything after a pause, at most 4 workers (author's rule, 2026-09-23). Safe to run several times.
cd "$(dirname "$0")"
if pgrep -f "resume_grid.py" > /dev/null; then echo "already running"; exit 0; fi
nohup python3 -u resume_grid.py --workers 4 > results_resume.log 2>&1 &
echo "resume started, pid $!, log results_resume.log"
