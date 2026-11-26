"""Rotate the combined GA results with the mouse, or save the view with --save."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path,
                        default=Path(__file__).with_name("ga_candidates.csv"))
    parser.add_argument("--save", type=Path, help="save the view instead of opening a window")
    args = parser.parse_args()
    temperature, mass, water = [], [], []
    with args.csv.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            temperature.append(float(row["pizza_c_20min"]))
            mass.append(float(row["box_mass_g"]))
            water.append(float(row["water_lost_g_20min"]))
    if not temperature:
        parser.error("The CSV contains no candidates.")

    fig = plt.figure(figsize=(10, 7), layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.scatter(temperature, mass, water, s=5, alpha=0.36,
               color="#146988", edgecolors="none", depthshade=False)
    ax.set(xlabel="Pizza temperature at 20 min (°C)", ylabel="Box mass (g)",
           zlabel="Water lost at 20 min (g)", title=f"Pizza box GA | {len(mass):,} accepted candidates")
    ax.view_init(elev=22, azim=-53)
    if args.save:
        fig.savefig(args.save, dpi=180, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
