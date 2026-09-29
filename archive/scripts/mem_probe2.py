import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np, torch, psutil
SCR="C:/Users/johna/AppData/Local/Temp/claude/D--Artificial-Nervous-System/efa12c9d-7852-4b73-9032-6dc08d47204d/scratchpad"
sys.path.insert(0, SCR+"/rsna_knee/raptor/cnn_kernel/push_dir")
import raptor_cnn_kernel as K
dev="cuda"
model,res=K.load_model(SCR+"/rsna_knee/raptor/raptor_ft_cnn336.pt", dev)
# ONE volume held in memory, reused every iter -> removes memmap page-cache confound
vols=np.load(SCR+"/rsna_knee/raptor/all_vols.npy", mmap_mode='r')
vol=np.array(vols[0]); mask=(vol.reshape(44,-1).sum(1)>0).astype('uint8')
proc=psutil.Process()
rss=lambda: proc.memory_info().rss/1e9
print(f"start CPU_RSS={rss():.3f}GB", flush=True)
for i in range(600):
    xw=K.eval_windows(vol, mask, k=K.K_EVAL, res=res, norm=K.NORM)
    p=K.infer_probs(model, xw, dev)
    if (i+1)%100==0: print(f"  {i+1}/600 CPU_RSS={rss():.3f}GB GPU_resv={torch.cuda.memory_reserved()/1e9:.2f}", flush=True)
print(f"END CPU_RSS={rss():.3f}GB  (flat => inference path clean; leak is in build_study/pydicom)", flush=True)
