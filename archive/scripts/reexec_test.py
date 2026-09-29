import os, sys, subprocess, numpy as np
def main():
    N, CHUNK = 23, 7
    if "--chunk" in sys.argv:
        k=sys.argv.index("--chunk"); lo,hi=int(sys.argv[k+1]),int(sys.argv[k+2])
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),f"_t_{lo}_{hi}.npy"), np.arange(lo,hi))
        print(f"  CHUNK {lo}:{hi} pid={os.getpid()}", flush=True); return
    script=os.path.abspath(__file__); d=os.path.dirname(script)
    full=np.full(N,-1); pids=set()
    for lo in range(0,N,CHUNK):
        hi=min(lo+CHUNK,N); out=os.path.join(d,f"_t_{lo}_{hi}.npy")
        r=subprocess.run([sys.executable,script,"--chunk",str(lo),str(hi)],capture_output=True,text=True,timeout=60)
        print(r.stdout.strip()); 
        if 'pid=' in r.stdout: pids.add(r.stdout.split('pid=')[1].strip())
        full[lo:hi]=np.load(out); os.remove(out)
    ok=all(full[i]==i for i in range(N))
    print(f"MERGE correct: {ok} | chunks: {(N+CHUNK-1)//CHUNK} | distinct child PIDs: {len(pids)} (each chunk = fresh process => RAM reclaimed)")
    print("REEXEC MECHANISM VALID" if ok and len(pids)==(N+CHUNK-1)//CHUNK else "FAILED")
if __name__=="__main__": main()
