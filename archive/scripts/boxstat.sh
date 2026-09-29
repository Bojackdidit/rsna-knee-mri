#!/bin/bash
# status check kept in a FILE so the process names never appear in an ssh command line
cd /workspace/knee
echo "merge procs : $(ps -eo cmd | grep -c "^python -u _merge_v11")"
echo "watcher     : $(ps -eo cmd | grep -c "^bash ./await_corpus_then_train.sh")"
echo "trainers    : $(ps -eo cmd | grep -c "^python -u finetune_box.py")"
echo "chunks done : $(grep -c "rows placed" merge_final.log 2>/dev/null) of 25"
echo "last        : $(grep "rows placed" merge_final.log 2>/dev/null | tail -1)"
echo "watcher log : $(tail -1 await_train.log 2>/dev/null)"
