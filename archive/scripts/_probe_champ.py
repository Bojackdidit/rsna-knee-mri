import torch, json
ck = torch.load("probe_outside/champ_fold0.pt", map_location="cpu", weights_only=False)
print("fingerprint:", json.dumps({k: str(v)[:60] for k, v in ck["fingerprint"].items()}, indent=1))
print("provenance :", json.dumps({k: str(v)[:70] for k, v in ck["provenance"].items()}, indent=1))
sd = ck["model"]; ks = list(sd)
print("state_dict tensors:", len(ks))
print("first:", ks[:4])
print("last :", ks[-4:])
print("head-ish:", [k for k in ks if any(t in k.lower() for t in ("head", "cls", "fc", "logit"))][:6])
