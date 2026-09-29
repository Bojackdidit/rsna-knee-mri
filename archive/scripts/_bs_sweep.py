import os, time, sys, torch, torch.nn as nn, timm
dev="cuda"; torch.backends.cudnn.benchmark=True
def bb(): return timm.create_model("convnext_small",pretrained=False,num_classes=0,in_chans=3,global_pool="avg").to(dev)
class Head(nn.Module):
    def __init__(s,ind,outd=8192,h=2048,b=256):
        super().__init__(); s.mlp=nn.Sequential(nn.Linear(ind,h),nn.GELU(),nn.Linear(h,h),nn.GELU(),nn.Linear(h,b))
        s.last=nn.utils.weight_norm(nn.Linear(b,outd,bias=False))
    def forward(s,x): return s.last(nn.functional.normalize(s.mlp(x),dim=-1))
sbb,tbb=bb(),bb(); sh,th=Head(768).to(dev),Head(768).to(dev)
sp=list(sbb.parameters())+list(sh.parameters()); tp=list(tbb.parameters())+list(th.parameters())
for p in tp: p.requires_grad=False
opt=torch.optim.AdamW(sp,lr=1e-4)
alloc=os.environ.get("PYTORCH_CUDA_ALLOC_CONF","(default)")
print(f"torch {torch.__version__} | alloc_conf={alloc}",flush=True)
for BS in [48,32,24,16]:
    try:
        g=[torch.rand(BS,3,224,224,device=dev) for _ in range(2)]; l=[torch.rand(BS,3,96,96,device=dev) for _ in range(6)]
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        def full():
            with torch.autocast(dev,dtype=torch.bfloat16):
                with torch.no_grad(): t_out=th(tbb(torch.cat(g,0)))
                s_out=sh(torch.cat([sbb(torch.cat(g,0)),sbb(torch.cat(l,0))],0))
                loss=-(torch.log_softmax(s_out.float()/0.1,-1)*torch.softmax(t_out.float()/0.04,-1).repeat(4,1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        ts=[]
        for i in range(4):
            t0=time.perf_counter(); full(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        peak=torch.cuda.max_memory_allocated()/1e9; ms=sorted(ts)[len(ts)//2]*1000
        print(f"  bs{BS:2d}: {ms:6.0f} ms/step  peak {peak:.1f}GB  {BS/(ms/1000):5.0f} samp/s",flush=True)
        del g,l; torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  bs{BS:2d}: OOM/ERR {str(e)[:60]}",flush=True); torch.cuda.empty_cache()
print("DONE",flush=True)
