"""Why does PF OA want the partner alone? Solo AUCs per arm, and the macro cost/benefit."""
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
LAB = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA',
       'Synovitis','Medial Cartilage','Lateral Cartilage','PF Cartilage','Effusion']
def load(t):
    d = np.load(f"gold_oof/raptor_gold_{t}.npz", allow_pickle=True); return d["ids"],d["pred"],d["truth"]
ia,pa,ta = load("SCRATCH_AUG2"); ib,pb,_ = load("CONVNEXT_E24")
common=[u for u in ia if u in set(ib)]
xa={u:i for i,u in enumerate(ia)}; xb={u:i for i,u in enumerate(ib)}
A=np.array([pa[xa[u]] for u in common]); B=np.array([pb[xb[u]] for u in common])
Y=np.array([ta[xa[u]] for u in common]); N=len(common)
def auc(p,f):
    y=Y[:,f].round().astype(int)
    return roc_auc_score(y,p) if len(set(y))>1 else float('nan')
print(f"{'finding':<20}{'champ':>8}{'partner':>9}{'blend.55':>10}{'partner-champ':>14}{'prev':>7}")
for f in range(12):
    ra=rankdata(A[:,f])/N; rb=rankdata(B[:,f])/N
    ca,cb=auc(A[:,f],f),auc(B[:,f],f); bl=auc(0.55*ra+0.45*rb,f)
    prev=Y[:,f].round().mean()
    print(f"{LAB[f]:<20}{ca:>8.4f}{cb:>9.4f}{bl:>10.4f}{cb-ca:>+14.4f}{prev:>7.2f}")
# macro with and without the PF OA override
def macro(wvec):
    s=[]
    for f in range(12):
        ra=rankdata(A[:,f])/N; rb=rankdata(B[:,f])/N
        a=auc(wvec[f]*ra+(1-wvec[f])*rb,f)
        if a==a: s.append(a)
    return float(np.mean(s))
base=[0.55]*12; ovr=[0.55]*12; ovr[6]=0.0
print(f"\nmacro shipped            {macro(base):.4f}")
print(f"macro with PF OA -> 0.00 {macro(ovr):.4f}   delta {macro(ovr)-macro(base):+.4f}")
print(f"\nboard noise floor is ~0.002; a delta under that cannot be read from one submit.")
