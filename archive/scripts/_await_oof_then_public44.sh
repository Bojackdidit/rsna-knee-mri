#!/bin/bash
# Run PUBLIC44 AFTER the OOF dump+assembly finishes, never alongside it. The OOF dump runs
# gate_eval.py (peak ~6.2 GB) which the release chain's finetune_box wait does NOT detect, so the
# two would collide on the 12.9 GB card. This gates on the OOF artifacts + gate_eval being gone.
set -u
cd "$(dirname "$0")"

gate_running () { powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -match 'gate_eval|_oof_validate' } | Measure-Object).Count" 2>/dev/null | tr -d '\r' | head -1; }

echo "=== PUBLIC44 gate armed $(date +%H:%M), waiting for OOF assembly ==="
# F4 is dumped last (fold order 0,1,2,3,4); its npz means the dump loop is on its final step.
while [ ! -f "oof/oof_LR96_F4.npz" ]; do sleep 180; done
# then wait for the assembly (gate_eval / _oof_validate) to fully exit and GPU to be idle of it
while [ "$(gate_running)" != "0" ]; do sleep 60; done
sleep 60
echo "=== OOF done, launching PUBLIC44 $(date +%H:%M) ==="
bash _release44_chain.sh
echo "=== PUBLIC44 chain returned $(date +%H:%M) ==="
