import os, sys, time, gc
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import numpy as np, torch, psutil
SCR="C:/Users/johna/AppData/Local/Temp/claude/D--Artificial-Nervous-System/efa12c9d-7852-4b73-9032-6dc08d47204d/scratchpad"
sys.path.insert(0, SCR+"/rsna_knee/raptor/cnn_kernel/push_dir")
import raptor_cnn_kernel as K   # import-safe eval half

dev="cuda"
model,res=K.load_model(SCR+"/rsna_knee/raptor/raptor_ft_cnn336.pt", dev)
# emulate the kernel's multi-gpu path effect as closely as single-GPU allows:
# (DataParallel needs >=2 devices; single-GPU test isolates whether the BASE loop leaks)
vols=np.load(SCR+"/rsna_knee/raptor/all_vols.npy", mmap_mode='r')
try: masks=np.load(SCR+"/rsna_knee/raptor/all_masks.npy")
except Exception: masks=None
N=min(600, vols.shape[0])
proc=psutil.Process()
def rss(): return proc.memory_info().rss/1e9
def gpu_a(): return torch.cuda.memory_allocated()/1e9
def gpu_r(): return torch.cuda.memory_reserved()/1e9
print(f"start: CPU_RSS={rss():.2f}GB GPU_alloc={gpu_a():.2f} GPU_resv={gpu_r():.2f}", flush=True)
t0=time.time()
for i in range(N):
    vol=np.asarray(vols[i])                       # (44,336,336) u8
    mask=masks[i] if masks is not None else (vol.reshape(44,-1).sum(1)>0).astype('uint8')
    xw=K.eval_windows(vol, mask, k=K.K_EVAL, res=res, norm=K.NORM)
    p=K.infer_probs(model, xw, dev)
    if (i+1)%50==0 or i+1==N:
        print(f"  {i+1}/{N} | CPU_RSS={rss():.2f}GB GPU_alloc={gpu_a():.2f} GPU_resv={gpu_r():.2f} | {(time.time()-t0):.0f}s | {(time.time()-t0)/(i+1)*1000:.0f}ms/study", flush=True)
print(f"END peak_GPU_resv={gpu_r():.2f}GB final_CPU_RSS={rss():.2f}GB", flush=True)
