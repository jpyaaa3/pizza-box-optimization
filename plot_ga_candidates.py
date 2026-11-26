"""Rotate the combined GA results with the mouse, or save the view with --save."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_points(path):
    label = path.stem
    if path.is_dir():
        settings = json.loads((path/"settings.json").read_text(encoding="utf-8"))
        weights = settings.get("rank_weights", dict(mass=1, temperature=1, drying=1))
        label = ":".join(f"{weights[k]:g}" for k in ("mass","temperature","drying"))
        label += f" | seed {settings['seed']}"
        path = path/"evaluations.csv"
    temperature, mass, water = [], [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if "status" in row:
                if row["status"] != "accepted":
                    continue
                temperature.append(float(row["pizza_c"]))
                mass.append(float(row["mass_g"]))
                water.append(3*(100-float(row["water_percent"])))
            else:
                temperature.append(float(row["pizza_c_20min"]))
                mass.append(float(row["box_mass_g"]))
                water.append(float(row["water_lost_g_20min"]))
    if not temperature:
        raise ValueError(f"No accepted candidates in {path}")
    return label, temperature, mass, water


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, help="combined CSV or GA run directories")
    parser.add_argument("--save", type=Path, help="save the view instead of opening a window")
    args = parser.parse_args()
    paths = args.inputs or [Path(__file__).with_name("ga_candidates.csv")]

    fig = plt.figure(figsize=(10, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    total = 0
    for i,path in enumerate(paths):
        label, temperature, mass, water = load_points(path)
        total += len(mass)
        ax.scatter(temperature, mass, water, s=16 if len(paths)>1 else 5, alpha=0.58,
                   marker=("o","^","s","D")[i%4], depthshade=False,
                   label=f"{label}  (n={len(mass):,})")
    if len(paths)>1:
        ax.legend()
    ax.set(xlabel="Pizza temperature at 20 min (°C)", ylabel="Box mass (g)",
           zlabel="Water lost at 20 min (g)", title=f"Pizza box GA | {total:,} accepted evaluations")
    ax.view_init(elev=22, azim=-53)
    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
