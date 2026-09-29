#!/usr/bin/env python3
"""Measure inference cost per study for each backbone, the way the kernel actually runs them.

The 3-study dry run says the ConvNeXt arm costs about 2.5x a CoAtNet arm, but a dry run is
dominated by warm-up and cannot be trusted to price a 1,322-study submission. This times the real
thing instead: K_EVAL windows at 384 through each backbone, steady state, warm-up discarded.
"""
import time, torch, timm

ARCHS = ["coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k",
         "convnext_base.fb_in22k_ft_in1k_384",
         "convnext_small.fb_in22k_ft_in1k_384",
         "convnext_tiny.fb_in22k_ft_in1k_384"]
K, RES, BS = 48, 384, 8
dev = "cuda:0"
print("K_EVAL %d windows at %d, batch %d\n" % (K, RES, BS))
base = None
for a in ARCHS:
    try:
        m = timm.create_model(a, pretrained=False, num_classes=0, in_chans=3, global_pool="avg")
    except Exception as e:
        print("  %-42s UNAVAILABLE (%s)" % (a, str(e)[:60])); continue
    m = m.eval().to(dev)
    x = torch.randn(BS, 3, RES, RES, device=dev)
    with torch.no_grad(), torch.autocast(dev, dtype=torch.bfloat16):
        for _ in range(6):                       # warm-up, discarded
            m(x)
        torch.cuda.synchronize()
        t0 = time.time()
        n = 0
        while n * BS < K * 6:                    # six studies' worth
            m(x); n += 1
        torch.cuda.synchronize()
        dt = time.time() - t0
    per_study = dt / 6.0
    if base is None:
        base = per_study
    print("  %-42s %6.1f ms/study  %.2fx" % (a, per_study * 1000, per_study / base))
    del m, x
    torch.cuda.empty_cache()
