import os, numpy as np
_LEAK=[]
def _winit(p): globals()['_P']=p
def _wtask(i):
    _LEAK.append(bytearray(4*1024*1024))     # 4MB leak/task
    return (i, np.full((2,4,4), i%250, np.uint8), os.getpid())
if __name__=='__main__':
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    N,RECYCLE,NW=40,8,2
    try: ctx=mp.get_context('forkserver')
    except Exception: ctx=mp.get_context('spawn')
    ex=ProcessPoolExecutor(max_workers=NW, mp_context=ctx, initializer=_winit, initargs=("x",), max_tasks_per_child=RECYCLE)
    futs={ex.submit(_wtask,i):i for i in range(N)}
    preds=[None]*N; pids=set()
    for fut in as_completed(futs):
        i,vol,pid=fut.result(); preds[i]=int(vol.flat[0]); pids.add(pid)
    ex.shutdown()
    ok=all(preds[i]==(i%250) for i in range(N))
    print(f"ctx={getattr(ctx,'_name','?')} | index-mapping OK: {ok} | distinct PIDs: {len(pids)} (>{NW} => recycling works)")
    print("MECHANISM VALID" if ok and len(pids)>NW else "MECHANISM FAILED")
