import os, numpy as np
_LEAK=[]                                   # simulate a per-task worker memory leak
def _winit(payload): globals()['_P']=payload
def _wtask(i):
    _LEAK.append(bytearray(8*1024*1024))   # leak 8MB/task (unbounded w/o recycle)
    import psutil
    vol=np.full((4,8,8), i%250, np.uint8)  # dummy volume encoding the index
    return (i, vol, os.getpid(), psutil.Process().memory_info().rss/1e6)
if __name__=='__main__':
    import torch, multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    m=torch.nn.Linear(8,8).cuda(); _=m(torch.randn(2,8).cuda())   # CUDA active in MAIN
    N,RECYCLE,NW=200,30,2
    try: ctx=mp.get_context('forkserver')
    except Exception: ctx=mp.get_context('spawn')
    ex=ProcessPoolExecutor(max_workers=NW, mp_context=ctx, initializer=_winit,
                           initargs=("x",), max_tasks_per_child=RECYCLE)
    futs={ex.submit(_wtask,i):i for i in range(N)}
    preds=[None]*N; pids=set(); mx=0.0
    for fut in as_completed(futs):
        i,vol,pid,rss=fut.result()
        t=torch.from_numpy(vol.astype('float32')).cuda(); _=t.mean().item()  # GPU op in main
        preds[i]=int(vol.flat[0]); pids.add(pid); mx=max(mx,rss)
    ex.shutdown()
    ok=all(preds[i]==(i%250) for i in range(N))
    print(f"ctx={getattr(ctx,'_name','?')}")
    print("INDEX-MAPPING correct:", ok)
    print(f"distinct worker PIDs: {len(pids)} over {N} tasks (recycle/{RECYCLE}; ~{N//RECYCLE*NW}+ = recycling works, 2 = NOT)")
    print(f"max worker RSS: {mx:.0f}MB  (bounded if << unbounded ~{N*8//NW}MB)")
    print("PATTERN VALID" if ok and len(pids)>NW and mx < 8*RECYCLE*3 else "PATTERN CHECK FAILED")
