"""Which proximity signal actually discriminates?

The recorded design says: for each unresolved target, take the nearest few
neighbours by the embedding of the paragraphs it appears in. Measured on the
real vault that produced 241 pairs, and the top ones were `circumburst medium`
against `isotropic energy` - two concepts that share nothing except a paper.

The hidden assumption is that mention context varies between concepts. In a
single-topic vault it does not: every concept appears in overlapping
paragraphs, so every context vector points the same way, and "nearest by
context" degenerates into "mentioned in the same document".

This script measures four candidate signals side by side rather than picking
one. Read the top pairs, not the scores - the scores on this model sit in a
narrow band whatever the answer.

    PYTHONPATH=. python3 -m sentinel.tests.field_test_proximity \\
        --vault ~/obsidian/Obsidian-1 --model ~/models/multilingual-e5-small \\
        --exclude agent_workspace
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..embed import E5Encoder, from_blob
from ..index import Index
from ..names import DEFER, resolve_pair


def load(idx):
    names, seeds = {}, set()
    for r in idx.db.execute(
            "SELECT target_key, MIN(display) d, MIN(resolved) res FROM links "
            "GROUP BY target_key"):
        names[r["target_key"]] = r["d"]
        if not r["res"]:
            seeds.add(r["target_key"])
    for r in idx.db.execute("SELECT page_key, path FROM files"):
        names.setdefault(r["page_key"], r["path"].rsplit("/", 1)[-1][:-3])
    return names, seeds


def context_vectors(idx, names):
    import numpy as np
    out = {}
    for key, display in names.items():
        rows = list(idx.db.execute(
            """SELECT v.vec FROM chunks c JOIN vectors v
                 ON v.stripped_hash = c.stripped_hash
               WHERE c.is_summary=0 AND c.content LIKE ?
                 AND (c.path IN (SELECT source FROM links WHERE target_key=?)
                      OR c.path IN (SELECT path FROM files WHERE page_key=?))
               LIMIT 5""", (f"%{display}%", key, key)))
        if rows:
            m = np.array([from_blob(r["vec"]) for r in rows], dtype=np.float32)
            v = m.mean(axis=0)
            n = np.linalg.norm(v)
            if n:
                out[key] = v / n
    return out


def top_pairs(vectors, names, seeds, k=10):
    import numpy as np
    keys = sorted(vectors)
    if len(keys) < 2:
        return []
    mat = np.array([vectors[k_] for k_ in keys], dtype=np.float32)
    pos = {k_: i for i, k_ in enumerate(keys)}
    scored = {}
    for seed in sorted(seeds & vectors.keys()):
        sims = mat @ vectors[seed]
        sims[pos[seed]] = -2.0
        for j in np.argsort(-sims)[:3]:
            a, b = sorted((seed, keys[j]))
            if resolve_pair(names[a], names[b]) != DEFER:
                continue
            scored[(a, b)] = max(scored.get((a, b), -2), float(sims[j]))
    return sorted(scored.items(), key=lambda x: -x[1])[:k]


def report(label, vectors, names, seeds):
    import numpy as np
    pairs = top_pairs(vectors, names, seeds)
    keys = sorted(vectors)
    if len(keys) > 1:
        mat = np.array([vectors[k] for k in keys], dtype=np.float32)
        sims = mat @ mat.T
        off = sims[~np.eye(len(keys), dtype=bool)]
        spread = f"similarity {off.min():.3f} to {off.max():.3f}, " \
                 f"median {float(np.median(off)):.3f}"
    else:
        spread = "too few vectors"
    print(f"\n=== {label} ===\n  {spread}")
    print(f"  {len(pairs)} pairs, top:")
    for (a, b), s in pairs:
        print(f"    {names[a]:<34} ? {names[b]:<34} {s:.3f}")
    if not pairs:
        print("    none")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vault", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--exclude", action="append", default=[])
    args = ap.parse_args()

    import numpy as np
    v = Path(args.vault).expanduser()
    idx = Index(v, v.parent / ".sentinel.db", exclude=args.exclude)
    idx.sync()
    names, seeds = load(idx)
    enc = E5Encoder(args.model)
    print(f"{len(names)} names, {len(seeds)} unresolved")

    # 1. context, exactly as the design records it
    ctx = context_vectors(idx, names)
    report("mention context (the recorded design)", ctx, names, seeds)

    # 2. context with the vault centroid removed. If every concept sits in the
    #    same document, the shared direction is the document - subtracting it
    #    leaves whatever is actually specific to each concept.
    if ctx:
        keys = sorted(ctx)
        mean = np.mean([ctx[k] for k in keys], axis=0)
        centred = {}
        for k in keys:
            d = ctx[k] - mean
            n = np.linalg.norm(d)
            if n > 1e-6:
                centred[k] = d / n
        report("mention context, vault centroid removed", centred, names, seeds)

    # 3. the NAME on its own. No context at all - two names that mean the same
    #    thing may simply read alike.
    keys = sorted(names)
    vecs = enc.encode([names[k] for k in keys], kind="passage")
    by_name = {k: np.asarray(v_, dtype=np.float32) for k, v_ in zip(keys, vecs)}
    report("the name alone", by_name, names, seeds)

    # 4. both, equally weighted.
    both = {}
    for k in keys:
        if k in ctx:
            v_ = by_name[k] + ctx[k]
            n = np.linalg.norm(v_)
            if n:
                both[k] = v_ / n
    report("name and context together", both, names, seeds)

    print("\nRead the PAIRS, not the numbers. Measured on this vault the name "
          "alone won\noutright, context scored 1.000 on everything because "
          "concepts share chunks,\nand name+context was worse than the name "
          "by itself. resolve_queue now\ngenerates from the name; this script "
          "is here to be re-run when the vault\nlooks different.")


if __name__ == "__main__":
    main()
