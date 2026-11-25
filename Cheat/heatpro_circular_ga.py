"""Axisymmetric FEM and rank-sum GA for a sealed circular top-hat box."""

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

for thread_setting in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[thread_setting] = "1"

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from scipy.stats import rankdata


BASE_MM = 150.10
PIZZA_RADIUS_MM = 150.0
PIZZA_HEIGHT_MM = 15.0
BOARD_MM = 3.0
BOARD_DENSITY = 200.0
PIZZA_AREA = math.pi * (0.150**2 + 2 * 0.150 * 0.015)
PIZZA_VOLUME = math.pi * 0.150**2 * 0.015
AMBIENT_C = 22.0
DURATION_S = 1200
MAX_WORKERS = min(61, os.cpu_count() or 1)


@dataclass(frozen=True)
class Hat:
    start_mm: float
    gap_mm: float
    rim_radius_mm: float

    @property
    def air_m3(self):
        return math.pi * (BASE_MM**2 * self.start_mm + self.rim_radius_mm**2 * self.gap_mm) * 1e-9 - PIZZA_VOLUME

    @property
    def mass_g(self):
        lower = (BASE_MM + BOARD_MM)**2 * self.start_mm
        upper = (self.rim_radius_mm + BOARD_MM)**2 * (self.gap_mm + 2 * BOARD_MM)
        cavity = BASE_MM**2 * self.start_mm + self.rim_radius_mm**2 * self.gap_mm
        return math.pi * (lower + upper - cavity) * BOARD_DENSITY * 1e-6


def axis_grid(breakpoints, spacing):
    cuts = np.unique(np.round(breakpoints, 10))
    pieces = [
        np.linspace(a, b, max(1, math.ceil((b - a) / spacing)) + 1)[:-1]
        for a, b in zip(cuts[:-1], cuts[1:])
    ]
    return np.r_[np.concatenate(pieces), cuts[-1]] * 1e-3


def saturation(celsius):
    return 610.78 * np.exp(17.2694 * celsius / (celsius + 237.29))


def temperature(hat, mesh_mm, dt):
    """Use P1 triangles on the meridian with exact 2πr-weighted integrals."""
    shoulder = hat.start_mm
    ceiling = shoulder + hat.gap_mm
    # Two elements across the 3 mm board even when the global spacing is larger.
    half = BOARD_MM / 2
    r = axis_grid([0, PIZZA_RADIUS_MM, BASE_MM, BASE_MM + half, BASE_MM + BOARD_MM,
                   hat.rim_radius_mm, hat.rim_radius_mm + half, hat.rim_radius_mm + BOARD_MM], mesh_mm)
    z_breaks = [-BOARD_MM, -half, 0, ceiling, ceiling + half, ceiling + BOARD_MM]
    if hat.rim_radius_mm > BASE_MM:
        z_breaks.extend([shoulder - BOARD_MM, shoulder - half, shoulder])
    z = axis_grid(z_breaks, mesh_mm)
    rmid = (r[1:] + r[:-1])[:, None] / 2
    zmid = (z[1:] + z[:-1])[None, :] / 2
    inside = (zmid > 0) & (zmid < ceiling * 1e-3) & (
        rmid < np.where(zmid < shoulder * 1e-3, BASE_MM, hat.rim_radius_mm) * 1e-3
    )
    outer = rmid < np.where(zmid < (shoulder - BOARD_MM) * 1e-3,
                           BASE_MM + BOARD_MM, hat.rim_radius_mm + BOARD_MM) * 1e-3
    solid = outer & ~inside
    ij = np.argwhere(solid)
    i, j = ij.T
    volume = np.sum(np.pi * (r[i + 1]**2 - r[i]**2) * (z[j + 1] - z[j]))
    if not math.isclose(volume * BOARD_DENSITY * 1000, hat.mass_g, rel_tol=1e-10):
        raise ValueError("Mesh volume disagrees with the analytic board mass.")

    rr, zz = np.meshgrid(r, z, indexing="ij")
    nodes = np.arange(rr.size).reshape(rr.shape)
    a, b, c, d = nodes[i, j], nodes[i + 1, j], nodes[i + 1, j + 1], nodes[i, j + 1]
    triangles = np.column_stack([np.r_[a, a], np.r_[b, c], np.r_[c, d]])
    points = np.column_stack((rr.ravel(), zz.ravel()))
    active, tris = np.unique(triangles, return_inverse=True)
    tris = tris.reshape(-1, 3)
    renumber = np.full(len(points), -1)
    renumber[active] = np.arange(len(active))
    pizza_id, air_id = len(active), len(active) + 1
    size = len(active) + 2

    p = points[triangles]
    q = np.roll(p, -1, axis=1)
    s = np.roll(p, 1, axis=1)
    area = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1]) -
            (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])) / 2
    gradients = np.stack((q[:, :, 1] - s[:, :, 1], s[:, :, 0] - q[:, :, 0]), axis=2) / (2 * area[:, None, None])
    radius = p[:, :, 0]
    volume_weight = 2 * np.pi * area
    k = 0.08 * (volume_weight * radius.mean(axis=1))[:, None, None] * (gradients @ gradients.transpose(0, 2, 1))
    diagonal = np.arange(3)
    radius_sum = radius.sum(axis=1)
    m = radius[:, :, None] + radius[:, None, :] + radius_sum[:, None, None]
    m[:, diagonal, diagonal] = 2 * (2 * radius + radius_sum[:, None])
    m *= (200 * 1400 * volume_weight / 60)[:, None, None]

    tri_rows, tri_cols = np.broadcast_arrays(tris[:, :, None], tris[:, None, :])
    k_rows, k_cols, k_values = [tri_rows.ravel()], [tri_cols.ravel()], [k.ravel()]
    ground, wet_edges, wet_weights = [], [], []
    for di, dj, left, right in (
        (-1, 0, a, d), (1, 0, b, c), (0, -1, a, b), (0, 1, d, c)
    ):
        ni, nj = i + di, j + dj
        neighbor_exists = (ni >= 0) & (ni < solid.shape[0]) & (nj >= 0) & (nj < solid.shape[1])
        ni = np.clip(ni, 0, solid.shape[0] - 1)
        nj = np.clip(nj, 0, solid.shape[1] - 1)
        boundary = ~(neighbor_exists & solid[ni, nj])
        endpoints = np.column_stack((left, right))[boundary]
        edge_xy = points[endpoints]
        r_end = edge_xy[:, :, 0]
        length = np.linalg.norm(edge_xy[:, 1] - edge_xy[:, 0], axis=1)
        scale = 2 * np.pi * length / 12
        edge_mass = np.empty((len(endpoints), 2, 2))
        edge_mass[:, 0, 0] = scale * (3 * r_end[:, 0] + r_end[:, 1])
        edge_mass[:, 1, 1] = scale * (r_end[:, 0] + 3 * r_end[:, 1])
        edge_mass[:, 0, 1] = edge_mass[:, 1, 0] = scale * r_end.sum(axis=1)
        edge_nodes = renumber[endpoints]
        is_inner = (neighbor_exists & inside[ni, nj])[boundary]
        is_ground = np.isclose(edge_xy[:, :, 1], -BOARD_MM * 1e-3).all(axis=1)
        is_contact = is_inner & np.isclose(edge_xy[:, :, 1], 0).all(axis=1) & (
            r_end.max(axis=1) <= PIZZA_RADIUS_MM * 1e-3 + 1e-12
        )
        ground.extend(edge_nodes[is_ground].ravel())
        exposed = is_inner & ~is_contact
        wet_edges.extend(edge_nodes[exposed])
        wet_weights.extend(edge_mass[exposed].sum(axis=2))

        def add_block(ids, values):
            rows, cols = np.broadcast_arrays(ids[:, :, None], ids[:, None, :])
            k_rows.append(rows.ravel())
            k_cols.append(cols.ravel())
            k_values.append(values.ravel())

        add_block(edge_nodes[~is_inner & ~is_ground], 10 * edge_mass[~is_inner & ~is_ground])
        for mask, lump, h in ((is_contact, pizza_id, 10), (exposed, air_id, 3)):
            panel = edge_mass[mask]
            load = panel.sum(axis=2)
            local = np.zeros((len(panel), 3, 3))
            local[:, :2, :2] = panel
            local[:, :2, 2] = local[:, 2, :2] = -load
            local[:, 2, 2] = load.sum(axis=1)
            add_block(np.column_stack((edge_nodes[mask], np.full(len(panel), lump))), h * local)

    lump_ids = np.array([pizza_id, air_id])
    add_block(lump_ids[None, :], (3 * PIZZA_AREA * np.array([[1, -1], [-1, 1]]))[None, :, :])
    stiffness = coo_matrix((np.concatenate(k_values), (np.concatenate(k_rows), np.concatenate(k_cols))),
                           shape=(size, size)).tocsc()
    capacity = coo_matrix((np.r_[m.ravel(), 2250, 1.2 * 1005 * hat.air_m3],
                           (np.r_[tri_rows.ravel(), lump_ids], np.r_[tri_cols.ravel(), lump_ids])),
                          shape=(size, size)).tocsc()
    free = np.setdiff1d(np.arange(size), ground)
    m_dt = capacity[free][:, free] / dt
    factor = splu((m_dt + stiffness[free][:, free]).tocsc())
    state = np.zeros(len(free))
    state[-2] = 90 - AMBIENT_C
    full = np.zeros(size)
    edge_nodes = np.asarray(wet_edges)
    weights = np.asarray(wet_weights)
    exposed_area = weights.sum(axis=1)
    weights /= exposed_area[:, None]
    times = np.linspace(0, DURATION_S, round(DURATION_S / dt) + 1)
    pizza = np.empty(len(times))
    air = np.empty(len(times))
    walls = np.empty((len(times), len(edge_nodes)))
    for step in range(len(times)):
        full[free] = state
        pizza[step], air[step] = state[-2:]
        walls[step] = (full[edge_nodes] * weights).sum(axis=1)
        if step + 1 < len(times):
            state = factor.solve(m_dt @ state)
    return times, pizza + AMBIENT_C, air + AMBIENT_C, walls + AMBIENT_C, exposed_area


def pizza_water_lost(hat, history):
    times, pizza, air, walls, areas = history
    vapor_r, transfer = 461.526, 1e-3
    initial_vapor = 0.5 * saturation(air[0]) * hat.air_m3 / (vapor_r * (air[0] + 273.15))
    wall_pressure = saturation(walls)
    wall_coeff = transfer * areas / (vapor_r * (273.15 + (walls + air[:, None]) / 2))
    dt = times[1] - times[0]

    def rates(t, y):
        i = min(int(t / dt), len(times) - 2)
        f = (t - times[i]) / dt
        ta = air[i] + f * (air[i + 1] - air[i])
        tp = pizza[i] + f * (pizza[i + 1] - pizza[i])
        pv = max(0, y[0]) * vapor_r * (ta + 273.15) / hat.air_m3
        evap = transfer * PIZZA_AREA / (vapor_r * (273.15 + (ta + tp) / 2)) * max(0, saturation(tp) - pv)
        wall_flux = (wall_coeff[i:i+2] * np.maximum(0, pv - wall_pressure[i:i+2])).sum(axis=1)
        cond = wall_flux[0] + f * (wall_flux[1] - wall_flux[0])
        return [evap - cond, evap]

    result = solve_ivp(rates, (0, DURATION_S), [initial_vapor, 0], t_eval=[DURATION_S],
                       method="Radau", rtol=1e-6, atol=1e-10, max_step=4)
    if not result.success:
        raise ValueError(result.message)
    vapor_kg, lost_kg = result.y[:, -1]
    if not (0 <= lost_kg <= 0.300 and -1e-10 <= vapor_kg <= initial_vapor + lost_kg + 1e-10):
        raise ValueError("Water balance returned a nonphysical mass.")
    return float(lost_kg * 1000)


def evaluate(task):
    index, generation, hat, mesh, dt = task
    row = dict(id=index, generation=generation, **asdict(hat), mass_g=hat.mass_g,
               air_l=hat.air_m3 * 1000, status="outside_mass")
    if not 83 <= hat.mass_g <= 249:
        return row
    history = temperature(hat, mesh, dt)
    row["pizza_c"] = float(history[1][-1])
    if row["pizza_c"] < 68:
        row["status"] = "below_temperature"
        return row
    lost = pizza_water_lost(hat, history)
    row.update(status="accepted", water_lost_g=lost, water_percent=100 * (300 - lost) / 300,
               water_fraction_percent=100 * (300 - lost) / (750 - lost))
    return row


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def objectives(row):
    return row["mass_g"], -row["pizza_c"], -row["water_lost_g"]


def rank(pool, count):
    if not pool:
        return []
    values = np.array([objectives(row) for row in pool])
    ranks = rankdata(values, method="average", axis=0)
    scores = ranks.sum(axis=1)
    order = np.lexsort((np.array([row["id"] for row in pool]), values[:, 0], scores))
    return [
        dict(pool[i], mass_rank=float(ranks[i, 0]), heat_rank=float(ranks[i, 1]),
             water_rank=float(ranks[i, 2]), rank_sum=float(scores[i]))
        for i in order[:count]
    ]


def archive_add(archive, candidate):
    values = np.array(objectives(candidate))
    if archive:
        others = np.array([objectives(row) for row in archive])
        if np.any((others <= values).all(axis=1) & (others < values).any(axis=1)):
            return
        dominated = (values <= others).all(axis=1) & (values < others).any(axis=1)
        archive[:] = [row for row, remove in zip(archive, dominated) if not remove]
    archive.append(candidate)


def run(options):
    rng = np.random.default_rng(options.seed)
    low = np.array([options.min_start, np.log(options.min_gap), BASE_MM + options.min_brim])
    high = np.array([options.max_start, np.log(options.max_gap), options.max_radius])
    extremes = list(product((0.0, 1.0), repeat=3))
    # The first candidates include a shallow, low start at both radius extremes.
    first = [extremes[i] for i in (0, 1, 4, 5, 2, 3, 6, 7)]
    saved, archive, seen = [], [], set()
    count = submitted = 0
    directory = options.out
    directory.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(options), out=str(directory), base_inner_radius_mm=BASE_MM,
                    board_thickness_mm=BOARD_MM, model="axisymmetric_fem")
    (directory / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    started = time.perf_counter()
    emit("started", seed=options.seed, workers=options.workers)
    stop_path = directory / "stop.request"
    fields = ("id", "generation", "start_mm", "gap_mm", "rim_radius_mm", "mass_g", "air_l",
              "pizza_c", "water_lost_g", "water_percent", "water_fraction_percent", "status")
    ranking_fields = (*fields, "mass_rank", "heat_rank", "water_rank", "rank_sum")

    def save(name, rows, columns):
        with (directory / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, columns)
            writer.writeheader()
            writer.writerows(rows)

    def encode(row):
        coordinates = np.array([row["start_mm"], np.log(row["gap_mm"]), row["rim_radius_mm"]])
        return (coordinates - low) / (high - low)

    def decode(unit):
        height, log_gap, radius = low + np.clip(unit, 0, 1) * (high - low)
        return Hat(float(height), float(np.exp(log_gap)), float(radius))

    with ProcessPoolExecutor(max_workers=options.workers) as executor:
        with (directory / "evaluations.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fields)
            writer.writeheader()
            for generation in range(options.generations + 1):
                target = options.population if generation == 0 else options.children
                children = []
                proposals = 0
                while len(children) < target and not stop_path.exists() and proposals < target * 100:
                    batch = []
                    while len(batch) < min(options.workers, target - len(children)) and proposals < target * 100:
                        proposals += 1
                        if generation == 0 and first:
                            unit = first.pop(0)
                        elif not saved or rng.random() < 0.25:
                            unit = rng.random(3)
                        else:
                            parent = saved[min(rng.integers(len(saved), size=3))]
                            unit = encode(parent)
                            indices = rng.choice(3, size=1 + int(rng.random() < 0.3), replace=False)
                            unit[indices] += rng.normal(0, rng.choice([0.035, 0.12, 0.30]), len(indices))
                        hat = decode(unit)
                        if hat.rim_radius_mm == BASE_MM:
                            # With no shoulder, only the total interior height defines the box.
                            key = (round(hat.start_mm + hat.gap_mm, 6),)
                        else:
                            key = tuple(round(x, 6) for x in asdict(hat).values())
                        if key not in seen:
                            seen.add(key)
                            submitted += 1
                            batch.append((submitted, generation, hat, options.mesh, options.dt))
                    for result in executor.map(evaluate, batch):
                        count += 1
                        writer.writerow(result)
                        stream.flush()
                        if result["status"] == "accepted":
                            children.append(result)
                            archive_add(archive, result)
                        emit("evaluation", count=count, generation=generation, accepted=len(children),
                             target=target, status=result["status"], elapsed_s=time.perf_counter() - started)
                pool = [dict((key, row[key]) for key in fields if key in row) for row in saved] + children
                saved = rank(pool, options.population)
                save("population.csv", saved, ranking_fields)
                save("pareto.csv", archive, fields)
                if saved:
                    (directory / "best.json").write_text(json.dumps(saved[0], indent=2), encoding="utf-8")
                emit("generation", generation=generation, count=count, archive=len(archive),
                     pool_size=len(pool), leaders=saved[:12], elapsed_s=time.perf_counter() - started)
                if stop_path.exists():
                    emit("stopped")
                    return
    if not saved:
        raise SystemExit("No design satisfies the mass and temperature limits.")
    emit("finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--children", type=int, default=40)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--workers", type=int, default=min(8, MAX_WORKERS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mesh", type=float, default=8.0, help="Maximum meridian spacing in mm")
    parser.add_argument("--dt", type=float, default=4.0, help="Heat time step in seconds; divisor of 1200")
    parser.add_argument("--min-start", type=float, default=15.10)
    parser.add_argument("--max-start", type=float, default=90.0)
    parser.add_argument("--min-gap", type=float, default=0.05)
    parser.add_argument("--max-gap", type=float, default=35.0)
    parser.add_argument("--min-brim", type=float, default=0.0)
    parser.add_argument("--max-radius", type=float, default=260.0)
    args = parser.parse_args()
    if not (args.population >= 1 and args.children >= 1 and args.generations >= 0
            and 1 <= args.workers <= MAX_WORKERS and (args.seed is None or args.seed >= 0)):
        parser.error("Invalid GA population, generation count, worker count, or seed.")
    numbers = (args.mesh, args.dt, args.min_start, args.max_start,
               args.min_gap, args.max_gap, args.min_brim, args.max_radius)
    if not all(math.isfinite(value) for value in numbers) or not (
        args.mesh > 0 and 0 < args.dt <= DURATION_S
        and math.isclose(round(DURATION_S / args.dt) * args.dt, DURATION_S, abs_tol=1e-8)
        and PIZZA_HEIGHT_MM < args.min_start < args.max_start and 0 < args.min_gap < args.max_gap
        and 0 <= args.min_brim < args.max_radius - BASE_MM
    ):
        parser.error("Invalid geometry bounds, mesh, or time step.")
    if args.seed is None:
        args.seed = int(np.random.SeedSequence().generate_state(1)[0])
    run(args)


if __name__ == "__main__":
    main()
