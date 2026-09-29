"""Isolate the Raptor-CNN bottleneck: (A) pure GPU step compute (synthetic data, real model),
(B) pure dataloader throughput (no GPU). Run standalone with the training STOPPED (needs VRAM)."""
import os, sys, time, torch
sys.argv = [sys.argv[0]]  # avoid arg parsing in imported module
import importlib.util
spec = importlib.util.spec_from_file_location("prc", r"D:/claude2/main_track/knee/pretrain_raptor_cnn.py")
prc = importlib.util.module_from_spec(spec)
# prevent main() from running on import: the file guards with if __name__=="__main__"
spec.loader.exec_module(prc)

dev = "cuda"
BS, NL = 48, 6
NCROPS = 2 + NL

print("=== (A) PURE GPU STEP (synthetic data, real convnext_small DINO) ===", flush=True)
bb = prc.build_backbone("convnext_small"); feat = bb.num_features
student = prc.MultiCropWrapper(bb, prc.DINOHead(feat, 8192)).to(dev)
teacher = prc.MultiCropWrapper(prc.build_backbone("convnext_small"), prc.DINOHead(feat, 8192)).to(dev)
teacher.load_state_dict(student.state_dict())
for p in teacher.parameters(): p.requires_grad = False
dino = prc.DINOLoss(8192, NCROPS).to(dev)
opt = torch.optim.AdamW(student.parameters(), lr=5e-4)
# synthetic crops: 2x(48,3,224,224) + 6x(48,3,96,96)
crops = [torch.rand(BS,3,224,224,device=dev) for _ in range(2)] + \
        [torch.rand(BS,3,96,96,device=dev) for _ in range(NL)]
torch.cuda.synchronize()
for i in range(12):
    t0=time.perf_counter()
    with torch.autocast(dev, dtype=torch.bfloat16):
        with torch.no_grad(): t_out=teacher(crops[:2])
        s_out=student(crops)
        loss=dino(s_out.float(), t_out.float(), 0)
    opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        m=0.996
        for ps,pt in zip(student.parameters(), teacher.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1-m)
    torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    if i>=2: print(f"  gpu step {i}: {dt*1000:.0f} ms", flush=True)
del student, teacher, dino, opt, crops
torch.cuda.empty_cache()

print("\n=== (B) PURE DATALOADER (no GPU, 24 workers) ===", flush=True)
from torch.utils.data import DataLoader
ROOT=r"C:/Users/johna/AppData/Local/Temp/claude/D--Artificial-Nervous-System/efa12c9d-7852-4b73-9032-6dc08d47204d/scratchpad/rsna_knee/raptor"
aug=prc.DinoAug(n_local=NL)
ds=prc.KneeWindows(ROOT, aug)
dl=DataLoader(ds,batch_size=BS,shuffle=True,num_workers=24,drop_last=True,
              collate_fn=prc.collate,pin_memory=True,persistent_workers=True,prefetch_factor=4)
it=iter(dl)
t0=time.perf_counter(); next(it); warm=time.perf_counter()-t0
print(f"  first batch (spawn+warm): {warm:.1f}s", flush=True)
t0=time.perf_counter(); n=20
for _ in range(n): b=next(it)
dt=time.perf_counter()-t0
print(f"  {n} batches: {dt:.1f}s = {dt/n*1000:.0f} ms/batch = {BS*n/dt:.0f} samples/s", flush=True)
print(f"\nVERDICT: if (A)<<(B), DATA-bound (fix loader). If (A)~=step-time, COMPUTE-bound (fix recipe).", flush=True)
