"""Run HeatPro's within-generation rank-sum search and stream progress as JSON."""

import argparse
import atexit
import csv
import json
import os
import secrets
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import rankdata

from box_generator import BOARD_DENSITY_KG_M3, BOARD_THICKNESS_MM, BoxDesign, InvalidGeometry, build_box, sample_box, write_obj
from moisture import moisture_retention, water_fraction_percent
from ring_box import (InvalidRingBox, Net, RingBox, RingDesign, build_ring_box, mutate_ring_box, sample_ring_box,
                      unfold, write_net, write_obj as write_ring_obj)
from thermal_fem import solve_thermal

RUNTIME = Path(__file__).resolve().parent / ".fem_runtime"
if RUNTIME.is_dir():
    sys.path.insert(0, str(RUNTIME))

import gmsh
from fem_benchmark import mesh_box
from ring_fem import mesh_ring_box, unit_normal

MAX_WORKERS = min(61, os.cpu_count() or 1)
DEFAULT_WORKERS = min(8, MAX_WORKERS)


@dataclass
class Evaluation:
    id: int
    box: object
    generation: int
    mass_g: float
    pizza_c: float
    water_percent: float
    net: Net | None = None

    @property
    def objectives(self):
        return self.mass_g, -self.pizza_c, self.water_percent

    @property
    def water_fraction_percent(self):
        return water_fraction_percent(self.water_percent)


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def fingerprint(box):
    if isinstance(box, RingBox):
        return "ring", tuple(map(tuple, np.round(box.points, 4))), box.faces
    points = box.cavity.points[box.cavity.vertices]
    return tuple(sorted(map(tuple, np.round(points, 4))))


def mutate(parent, rng):
    if isinstance(parent.box, RingBox):
        return mutate_ring_box(parent.box, rng)
    design = parent.box.design
    angles = list(design.base_angles_deg)
    distances = list(design.base_distances_mm)
    points = list(design.upper_points_mm)
    move = rng.random()
    if move < 0.50:
        active = parent.box.cavity.vertices - len(angles)
        index = rng.choice(active[active >= 0])
        direction = rng.uniform(0, 2 * np.pi)
        horizontal = rng.uniform(5, 15) * np.array([np.cos(direction), np.sin(direction)])
        dz = rng.choice([-1, 1]) * rng.uniform(3, 8)
        points[index] = tuple(np.asarray(points[index]) + np.r_[horizontal, dz])
    elif move < 0.75:
        index = rng.integers(len(distances))
        distances[index] = max(151.0, distances[index] + rng.choice([-1, 1]) * rng.uniform(2, 6))
    elif move < 0.90:
        index = rng.integers(len(angles))
        angles[index] = (angles[index] + rng.choice([-1, 1]) * rng.uniform(3, 8)) % 360
    elif rng.random() < 0.5:
        if len(points) <= 1 or (len(points) < 16 and rng.random() < 0.5):
            points.append(tuple(rng.uniform((-211, -211, 18), (211, 211, 101))))
        else:
            points.pop(rng.integers(len(points)))
    else:
        if len(angles) <= 3 or (len(angles) < 12 and rng.random() < 0.5):
            angles.append(float(rng.uniform(0, 360)))
            distances.append(float(rng.uniform(152, 178)))
        else:
            index = rng.integers(len(angles))
            angles.pop(index)
            distances.pop(index)
    return build_box(BoxDesign(tuple(angles), tuple(distances), tuple(points)), clearance_mm=1.0)


def rank_population(pool, size):
    if not pool:
        return []
    ranks = rankdata([member.objectives for member in pool], method="average", axis=0)
    scores = ranks.sum(axis=1)
    features = np.asarray([
        (member.box.floor_area_m2, member.box.nonfloor_area_m2, member.box.air_volume_m3,
         member.box.rings[-1].height_mm if isinstance(member.box, RingBox)
         else np.ptp(member.box.cavity.points[member.box.cavity.vertices, 2])) for member in pool
    ])
    features = (features - features.min(axis=0)) / np.maximum(np.ptp(features, axis=0), 1e-12)
    selected = []
    for score in np.unique(scores):
        group = list(np.flatnonzero(scores == score))
        while group and len(selected) < size:
            index = group[0]
            if selected and len(group) > 1:
                difference = features[group, None, :] - features[selected, :]
                distance = (difference * difference).sum(axis=2).min(axis=1)
                index = group[int(distance.argmax())]
            selected.append(index)
            group.remove(index)
        if len(selected) >= size:
            break
    return [(pool[i], ranks[i].tolist(), float(scores[i])) for i in selected]


def archive_candidate(archive, member):
    values = np.asarray(member.objectives)
    if any(np.all(previous.objectives <= values) for previous in archive):
        return
    archive[:] = [previous for previous in archive if not np.all(values <= previous.objectives)]
    archive.append(member)


def row(member, ranks=None, score=None):
    data = {"id": member.id, "mass_g": member.mass_g, "pizza_c": member.pizza_c,
            "water_percent": member.water_percent, "water_fraction_percent": member.water_fraction_percent,
            "generation": member.generation}
    if ranks is not None:
        data.update(zip(("mass_rank", "heat_rank", "water_rank", "rank_sum"), [*ranks, score]))
    return data


def snapshot(directory, selected, archive):
    for name, members in (("population", selected), ("pareto", [(member, None, None) for member in archive])):
        path = directory / f"{name}.csv"
        tmp = path.with_suffix(".tmp")
        columns = ["id", "mass_g", "pizza_c", "water_percent", "water_fraction_percent", "generation", "mass_rank", "heat_rank",
                   "water_rank", "rank_sum", "design"]
        with tmp.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for member, ranks, score in members:
                writer.writerow(dict(row(member, ranks, score), design=json.dumps(asdict(member.box.design))))
        tmp.replace(path)
    if selected:
        best = selected[0][0]
        obj = directory / "best.obj.tmp"
        if isinstance(best.box, RingBox):
            pattern = directory / "best_net.svg.tmp"
            write_ring_obj(best.box, obj)
            write_net(best.box, best.net, pattern)
            pattern.replace(directory / "best_net.svg")
        else:
            write_obj(best.box, obj)
        obj.replace(directory / "best.obj")


def preview(member):
    if isinstance(member.box, RingBox):
        box = member.box
        triangles, directions = [], []
        for face in box.faces:
            normal = unit_normal(box.points[list(face)]).tolist()
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])
                directions.append(normal)
        return {"points": box.points.tolist(), "faces": triangles, "normals": directions}
    hull = member.box.outer
    points = hull.points[hull.vertices]
    index_by_vertex = {vertex: index for index, vertex in enumerate(hull.vertices)}
    faces = [[index_by_vertex[vertex] for vertex in face] for face in hull.simplices]
    return {"points": points.tolist(), "faces": faces, "normals": hull.equations[:, :3].tolist()}


def init_worker():
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    atexit.register(gmsh.finalize)


def evaluate_design(design, unfold_seed, preferred_net=None):
    begin = perf_counter()
    result = {"status": "accepted"}
    net = None
    stage = "geometry"
    try:
        if isinstance(design, RingDesign):
            box = build_ring_box(design.sides, design.rings, design.bands)
            stage = "net"
            net = unfold(box, np.random.default_rng(unfold_seed), preferred=preferred_net)
            if net is None:
                return dict(status="net_failed", detail="No one-piece, nonoverlapping net found.",
                            seconds=perf_counter() - begin), None
            result["mass_g"] = net.mass_g
            if not 83 <= net.mass_g <= 249:
                return dict(result, status="outside_mass", seconds=perf_counter() - begin), None
            stage = "mesh"
            mesh = mesh_ring_box(box, 8)
        else:
            box = build_box(design, clearance_mm=1.0)
            result["mass_g"] = box.mass_g
            stage = "mesh"
            mesh = mesh_box(box, 8)
        stage = "thermal"
        thermal = solve_thermal(mesh, dt_s=4, wall_sample_s=4)
        result["pizza_c"] = float(thermal.pizza_c[-1])
        if result["pizza_c"] < 68:
            result["status"] = "below_temperature"
        else:
            stage = "moisture"
            retention = moisture_retention(thermal, box.air_volume_m3)
            result["water_percent"] = retention
            result["water_fraction_percent"] = water_fraction_percent(retention)
    except Exception as error:
        if stage != "mesh" and not isinstance(error, (RuntimeError, ValueError, ArithmeticError)):
            raise
        result.update(status=f"{stage}_failed", detail=str(error))
    result["seconds"] = perf_counter() - begin
    return result, net if result["status"] == "accepted" else None


def run(directory, population, children, generations, seed, workers=DEFAULT_WORKERS, shape="ring"):
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "settings.json").write_text(json.dumps({"population": population, "children": children,
        "generations": generations, "seed": seed, "workers": workers, "mesh_mm": 8, "dt_s": 4, "shape": shape,
        "thermal_model": "layered_panels" if shape == "ring" else "cad_solid"}, indent=2),
        encoding="utf-8")
    rng = np.random.default_rng(seed)
    selected, archive = [], []
    seen = set()
    submitted = completed = 0
    started = perf_counter()
    emit("started", seed=seed, workers=workers, shape=shape, path=str(directory), total=population + children * generations)
    sample = sample_ring_box if shape == "ring" else sample_box
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    stop_file = directory / "stop.request"
    columns = ["id", "generation", "mass_g", "pizza_c", "water_percent", "water_fraction_percent",
               "status", "seconds", "detail", "design"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"),
                             initializer=init_worker) as executor:
        with (directory / "evaluations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for generation in range(generations + 1):
                pool = [member for member, _, _ in selected]
                target = population if generation == 0 else children
                accepted = proposals = fresh_accepted = fresh_pending = 0
                fresh_target = target if not selected else max(1, round(children * 0.10))
                proposal_limit = max(240, 40 * target)
                pending = deque()
                while True:
                    while (len(pending) < min(workers, target - accepted)
                           and proposals < proposal_limit and not stop_file.exists()):
                        proposals += 1
                        fresh = fresh_accepted + fresh_pending < fresh_target or proposals > 20 * target
                        preferred_net = None
                        try:
                            if fresh:
                                box = sample(rng)
                            else:
                                parent_index = min(rng.integers(len(selected), size=3))
                                parent = selected[parent_index][0]
                                box = mutate(parent, rng)
                                if parent.net is not None and box.faces == parent.box.faces:
                                    preferred_net = (parent.net.hinges,
                                                     {edge: tab.face for edge, tab in parent.net.tabs.items()})
                        except (InvalidGeometry, InvalidRingBox):
                            continue
                        key = fingerprint(box)
                        ring = isinstance(box, RingBox)
                        if ring:
                            bare_mass_g = box.face_areas_m2.sum() * BOARD_THICKNESS_MM * BOARD_DENSITY_KG_M3
                            mass_possible = bare_mass_g <= 249
                        else:
                            mass_possible = 83 <= box.mass_g <= 249
                        if key in seen or not mass_possible:
                            continue
                        if stop_file.exists():
                            break
                        seen.add(key)
                        submitted += 1
                        unfold_seed = (seed + submitted * 0x9E3779B97F4A7C15) % 2**64
                        future = executor.submit(evaluate_design, box.design, unfold_seed, preferred_net)
                        pending.append((submitted, box, fresh, future))
                        fresh_pending += fresh
                    if not pending:
                        break
                    candidate_id, box, fresh, future = pending.popleft()
                    fresh_pending -= fresh
                    result, net = future.result()
                    completed += 1
                    record = {"id": candidate_id, "generation": generation,
                              "design": json.dumps(asdict(box.design)), **result}
                    if result["status"] == "accepted":
                        member = Evaluation(candidate_id, box, generation, result["mass_g"],
                                            result["pizza_c"], result["water_percent"], net)
                        pool.append(member)
                        archive_candidate(archive, member)
                        accepted += 1
                        fresh_accepted += fresh
                    writer.writerow(record)
                    stream.flush()
                    emit("evaluation", count=completed, generation=generation, accepted=accepted, target=target,
                         status=record["status"], elapsed_s=perf_counter() - started)
                selected = rank_population(pool, population)
                snapshot(directory, selected, archive)
                emit("generation", generation=generation, archive=len(archive), count=completed, pool_size=len(pool),
                     accepted=accepted, target=target, elapsed_s=perf_counter() - started,
                     leaders=[dict(row(member, ranks, score), geometry=preview(member))
                              for member, ranks, score in selected[:12]])
                if stop_file.exists():
                    emit("stopped", elapsed_s=perf_counter() - started)
                    return
                if accepted < target:
                    raise RuntimeError(f"Generation {generation}: only {accepted}/{target} feasible candidates "
                                       f"after {proposals} proposals. Partial results were saved.")
    emit("finished", elapsed_s=perf_counter() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--children", type=int, default=40)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--shape", choices=("ring", "convex"), default="ring")
    args = parser.parse_args()
    if min(args.population, args.children) < 1 or args.generations < 0 or (args.seed is not None and args.seed < 0):
        parser.error("Population and children must be positive. Generations and seed must be nonnegative.")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"Use between 1 and {MAX_WORKERS} CPU workers.")
    run(args.out.resolve(), args.population, args.children, args.generations,
        secrets.randbits(32) if args.seed is None else args.seed, args.workers, args.shape)


if __name__ == "__main__":
    main()
