def legacy_group_members():
    paths = find_legacy_bundles()
    if not paths:
        log('no legacy bundle attached; blending skipped')
        return {}
    out_ms, key = ([], None)
    for p in paths:
      try:
        b = torch.load(p, map_location='cpu', weights_only=False)
        folds = b.get('fold_states') or []
        b_slots = [tuple(s)[0] for s in b.get('slots', SLOTS)]
        if list(b.get('targets', TARGETS)) != TARGETS or b_slots != [s[0] for s in SLOTS]:
            log(f'legacy bundle {p.name}: target/slot contract differs; skipped')
            continue
        gr, n_gr = (int(b.get('group', 3)), int(b.get('n_group', 3)))
        variant = str(b.get('model_variant', 'dinov2-small')).split('-')[-1]
        key = json.dumps({'img': int(b.get('img', 224)), 'group': gr, 'slices': gr * n_gr, 'crop_mm': 160.0, 'band': [0.2, 0.8], 'rules': RULES_LEGACY, 'slots': [s[0] for s in SLOTS]}, sort_keys=True)
        ms = [{'id': f"legacy-{p.stem.split('_')[-1]}-f{f.get('fold', k)}", 'fold': f.get('fold', k), 'state': f['state_dict'], 'holdout': None, 'weight': LEGACY_WEIGHT, 'target_weight': [LEGACY_MEMBER_WEIGHT_BY_TARGET.get(t, 0.0) / max(len(paths), 1) for t in TARGETS], 'pixel_group': key, 'config': {'unfreeze_last': 6, 'variant': 'base' if variant == 'base' else 'small', 'pool': 'cls_mean_focal', 'prior': True}} for k, f in enumerate(folds)]
        out_ms.extend(ms)
        log(f'legacy bundle {p.name}: {len(ms)} fold(s) joined')
      except Exception as exc:
        log(f'legacy bundle {p.name} unusable ({type(exc).__name__}: {exc}); skipped')
    if out_ms:
        log(f'legacy family: {len(out_ms)} member(s) from {len(paths)} run(s); '
            f'per-member target weights divided by {len(paths)} so the family share is unchanged')
    return {key: out_ms} if out_ms and key else {}
