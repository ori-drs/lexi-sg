#!/usr/bin/env python3
"""Summarise run_ate.sh results next to the paper's values (Tables II and III)."""

import argparse
import json
import os

PAPER = {
    "hm3d": {"00824": 0.343, "00829": 0.143, "00843": 0.628, "00847": 0.461,
             "00873": None, "00877": 0.554, "00890": 0.712},
    "aod": {"ff-1": 0.166, "ff-2": 0.277, "ff-3": 0.277,
            "gf-1": 0.647, "gf-2": 0.201, "gf-3": 0.262},
}


def fmt(v):
    return f"{v:8.3f}" if v is not None else f"{'X':>8s}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root", help="OUT_ROOT of run_ate.sh")
    ap.add_argument("--dataset", choices=sorted(PAPER), default=None, help="adds the paper's values")
    args = ap.parse_args()

    paper = PAPER.get(args.dataset, {})
    variants = sorted(d for d in os.listdir(args.out_root)
                      if os.path.isdir(os.path.join(args.out_root, d)) and d != "logs")
    for variant in variants:
        vdir = os.path.join(args.out_root, variant)
        scenes = sorted(set(os.listdir(vdir)) | set(paper))
        print(f"{variant}")
        print(f"  {'scene':8s} {'frames':>6s} {'sim3':>8s} {'se3':>8s} {'paper':>8s}")
        sim3, ref = [], []
        for scene in scenes:
            f = os.path.join(vdir, scene, "ate.json")
            if not os.path.exists(f):
                print(f"  {scene:8s} {'missing':>6s} {'':8s} {'':8s} {fmt(paper.get(scene))}")
                continue
            r = json.load(open(f))
            s, e = r["ate_trans_sim3"]["rmse"], r["ate_trans_se3"]["rmse"]
            print(f"  {scene:8s} {r['frames']:6d} {s:8.3f} {e:8.3f} {fmt(paper.get(scene))}")
            sim3.append(s)
            if paper.get(scene) is not None:
                ref.append((s, paper[scene]))
        if sim3:
            print(f"  {'mean':8s} {len(sim3):6d} {sum(sim3) / len(sim3):8.3f}")
        if ref:
            print(f"  {'mean*':8s} {len(ref):6d} {sum(s for s, _ in ref) / len(ref):8.3f} {'':8s} "
                  f"{sum(p for _, p in ref) / len(ref):8.3f}")
            print("  * scenes with a paper value only")
        print()


if __name__ == "__main__":
    main()
