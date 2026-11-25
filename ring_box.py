"""Generate polygon-ring pizza boxes and search for one-piece cutting patterns."""

import argparse
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from box_generator import BOARD_DENSITY_KG_M3, BOARD_THICKNESS_MM, PIZZA_HEIGHT_MM, PIZZA_RADIUS_MM


EPS = 1e-7  # mm for planar clearance and projected overlap tests


class InvalidRingBox(ValueError):
    """The proposed rings do not bound a usable pizza cavity."""


@dataclass(frozen=True)
class Ring:
    radius_mm: float
    height_mm: float
    angle_deg: float = 0.0


@dataclass(frozen=True)
class Band:
    shift: int
    flip_diagonal: bool


@dataclass(frozen=True)
class RingDesign:
    sides: int
    rings: tuple[Ring, ...]
    bands: tuple[Band, ...]


@dataclass(frozen=True)
class RingBox:
    points: np.ndarray
    faces: tuple[tuple[int, ...], ...]
    sides: int
    rings: tuple[Ring, ...]
    bands: tuple[Band, ...]
    cavity_volume_mm3: float
    convex_deficit: float

    @property
    def design(self):
        return RingDesign(self.sides, self.rings, self.bands)

    @cached_property
    def face_areas_m2(self):
        areas = []
        for face in self.faces:
            p = self.points[list(face)]
            areas.append(np.linalg.norm(np.cross(p[1:-1] - p[0], p[2:] - p[0]).sum(axis=0)) * 0.5e-6)
        return np.asarray(areas)

    @property
    def floor_area_m2(self):
        return self.face_areas_m2[0]

    @property
    def nonfloor_area_m2(self):
        return self.face_areas_m2[1:].sum()

    @property
    def air_volume_m3(self):
        return (self.cavity_volume_mm3 - np.pi * PIZZA_RADIUS_MM**2 * PIZZA_HEIGHT_MM) * 1e-9


@dataclass(frozen=True)
class Panel:
    xy: np.ndarray
    low: np.ndarray
    high: np.ndarray
    axes: np.ndarray

    @classmethod
    def from_xy(cls, xy):
        edges = np.roll(xy, -1, axis=0) - xy
        normals = np.column_stack((-edges[:, 1], edges[:, 0]))
        return cls(xy, xy.min(axis=0), xy.max(axis=0), normals / np.linalg.norm(normals, axis=1)[:, None])


@dataclass(frozen=True)
class Tab:
    face: int
    panel: Panel


@dataclass(frozen=True)
class Net:
    panels: dict[int, Panel]
    hinges: frozenset[tuple[int, int]]
    tabs: dict[tuple[int, int], Tab]
    mass_g: float


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def area2(xy):
    return 0.5 * abs(cross2(xy, np.roll(xy, -1, axis=0)).sum())


def wedge_stays_ordered(p0, p1, q0, q1):
    """Check the signed cross-product for every interior cut of a linear band."""
    dp, dq = p1 - p0, q1 - q0
    a = cross2(dp, dq)
    b = cross2(dp, q0) + cross2(p0, dq)
    c = cross2(p0, q0)
    if min(c, a + b + c) < -EPS or (a / 4 + b / 2 + c) <= EPS:
        return False
    if a > EPS:
        t = -b / (2 * a)
        if 1e-9 < t < 1 - 1e-9 and (a * t + b) * t + c <= EPS:
            return False
    return True


def clip_wall_at_height(face, z):
    result = []
    for a, b in zip(face, np.roll(face, -1, axis=0)):
        if a[2] <= z:
            result.append(a[:2])
        if (a[2] < z < b[2]) or (b[2] < z < a[2]):
            t = (z - a[2]) / (b[2] - a[2])
            result.append(a[:2] + t * (b[:2] - a[:2]))
    return np.array(result)


def minimum_radius(polygon):
    if not len(polygon):
        return np.inf
    end = np.roll(polygon, -1, axis=0)
    if len(polygon) >= 3 and area2(polygon) > EPS:
        turns = cross2(polygon, end)
        if np.all(turns >= -EPS) or np.all(turns <= EPS):
            return 0.0
    edges = end - polygon
    square = np.einsum("ij,ij->i", edges, edges)
    t = np.clip(-np.einsum("ij,ij->i", polygon, edges) / np.maximum(square, EPS**2), 0, 1)
    return float(np.linalg.norm(polygon + t[:, None] * edges, axis=1).min())


def build_ring_box(sides, rings, bands, *, clearance_mm=1.0):
    """Build an oriented, thin cavity boundary with flat caps and triangular walls."""
    if sides < 3 or len(rings) < 2 or len(bands) != len(rings) - 1:
        raise InvalidRingBox("Use at least three sides and one connection for each adjacent ring pair.")
    if not np.isfinite(clearance_mm) or clearance_mm < 0:
        raise InvalidRingBox("Clearance must be nonnegative and finite.")
    values = np.array([(r.radius_mm, r.height_mm, r.angle_deg) for r in rings], dtype=float)
    radii, depths, orientations = values.T
    if not np.isfinite(values).all() or np.any(radii <= EPS):
        raise InvalidRingBox("Rings need finite coordinates and positive radii.")
    top_clearance = PIZZA_HEIGHT_MM + clearance_mm
    if depths[0] != 0 or np.any(np.diff(depths) <= EPS) or depths[-1] < top_clearance:
        raise InvalidRingBox("Start at z=0 and increase ring heights until the lid clears the pizza.")
    angles = np.deg2rad(orientations[:, None]) + 2 * np.pi * np.arange(sides) / sides
    points = np.stack((radii[:, None] * np.cos(angles), radii[:, None] * np.sin(angles),
                       np.broadcast_to(depths[:, None], angles.shape)), axis=-1).reshape(-1, 3)
    faces = [tuple(reversed(range(sides)))]
    for layer, band in enumerate(bands):
        lo, hi = layer * sides, (layer + 1) * sides
        for i in range(sides):
            j = (i + band.shift) % sides
            a, b = lo + i, lo + (i + 1) % sides
            c, d = hi + (j + 1) % sides, hi + j
            if band.flip_diagonal:
                spine = (a, c)
                panels = ((a, b, c), (a, c, d))
            else:
                spine = (b, d)
                panels = ((a, b, d), (b, c, d))
            if i == 0:
                p, q = spine
                xy = points[:, :2]
                if (not wedge_stays_ordered(xy[a], xy[d], xy[p], xy[q])
                        or not wedge_stays_ordered(xy[p], xy[q], xy[b], xy[c])):
                    raise InvalidRingBox("A connection crosses itself within a ring band.")
            for face in panels:
                clipped = clip_wall_at_height(points[list(face)], top_clearance)
                if minimum_radius(clipped) < PIZZA_RADIUS_MM + clearance_mm - EPS:
                    raise InvalidRingBox("A folded wall enters the pizza clearance cylinder.")
            faces.extend(panels)
    faces.append(tuple(range(len(points) - sides, len(points))))
    volume = 0.0
    for face in faces:
        p = points[list(face)]
        volume += np.sum(np.cross(p[1:-1], p[2:]) @ p[0]) / 6
    if volume <= 0:
        raise InvalidRingBox("The proposed rings have no positive interior volume.")
    deficit = max(0.0, 1 - volume / ConvexHull(points).volume)
    return RingBox(points, tuple(faces), sides, tuple(rings), tuple(bands), volume, deficit)


def face_edges(face):
    return zip(face, (*face[1:], face[0]))


def edge_key(a, b):
    return (min(a, b), max(a, b))


def panel_coordinates(points):
    first, second = points[1:3] - points[0]
    axis = first / np.linalg.norm(first)
    normal = np.cross(first, second)
    normal /= np.linalg.norm(normal)
    return (points - points[0]) @ np.stack((axis, np.cross(normal, axis)), axis=1)


def align_panel(local, child, parent, ids, edge):
    ia, ib = (ids[child][vertex] for vertex in edge)
    source_a, source_b = local[child][[ia, ib]]
    target_a, target_b = parent
    direction = source_b - source_a
    destination = target_b - target_a
    length = np.linalg.norm(direction)
    source_unit, target_unit = direction / length, destination / length
    source_axes = np.stack((source_unit, [-source_unit[1], source_unit[0]]), axis=1)
    target_axes = np.stack((target_unit, [-target_unit[1], target_unit[0]]), axis=0)
    return Panel.from_xy((local[child] - source_a) @ source_axes @ target_axes + target_a)


def overlaps(a, b):
    """SAT for convex panels; contact along a boundary has zero area."""
    if np.any(np.minimum(a.high, b.high) - np.maximum(a.low, b.low) <= EPS):
        return False
    axes = np.concatenate((a.axes, b.axes))
    p, q = a.xy @ axes.T, b.xy @ axes.T
    overlap = np.minimum(p.max(axis=0), q.max(axis=0)) - np.maximum(p.min(axis=0), q.min(axis=0))
    return bool(np.all(overlap > EPS))


def make_tabs(panels, incidence, hinges, ids, width_mm, rng, preferred_tabs=None):
    occupied = list(panels.values())
    tabs = {}
    for edge in rng.permutation(list(incidence)):
        edge = tuple(edge)
        if edge in hinges:
            continue
        choices = rng.permutation(incidence[edge]).tolist()
        if preferred_tabs is not None:
            choices.sort(key=lambda entry: entry[0] != preferred_tabs.get(edge))
        for face, a, b in choices:
            start, end = panels[face].xy[[ids[face][a], ids[face][b]]]
            length = np.linalg.norm(end - start)
            inset = width_mm * 0.45
            if length <= 4 * inset + EPS:
                continue
            u = (end - start) / length
            out = np.array([u[1], -u[0]])
            base_a, base_b = start + inset * u, end - inset * u
            tab = Panel.from_xy(np.array([base_a, base_b, base_b - inset * u + width_mm * out,
                                         base_a + inset * u + width_mm * out]))
            if not any(overlaps(tab, panel) for panel in occupied):
                tabs[edge] = Tab(int(face), tab)
                occupied.append(tab)
                break
        else:
            return None
    return tabs


def unfold_strips(box, local, ids):
    """Try one edge-connected panel strip per base side, then attach the lid."""
    panels = {0: Panel.from_xy(local[0])}
    hinges = set()

    def attach(parent, child):
        edge = tuple(sorted(ids[parent].keys() & ids[child].keys()))
        parent_xy = panels[parent].xy[[ids[parent][vertex] for vertex in edge]]
        candidate = align_panel(local, child, parent_xy, ids, edge)
        if any(overlaps(candidate, panel) for panel in panels.values()):
            return False
        panels[child] = candidate
        hinges.add(edge)
        return True

    ends = []
    for sector in range(box.sides):
        parent = 0
        for layer, band in enumerate(box.bands):
            lower = 1 + 2 * (layer * box.sides + sector)
            upper = lower + 1
            if not attach(parent, lower) or not attach(lower, upper):
                return None
            parent = upper
            sector = (sector + band.shift) % box.sides
        ends.append(parent)
    lid = len(box.faces) - 1
    for parent in ends:
        if attach(parent, lid):
            return panels, hinges
    return None


def unfold(box, rng, *, attempts=32, tab_mm=4.0, preferred=None):
    """Try bounded randomized hinge trees. Return None if no valid net is found."""
    if attempts < 1 or not np.isfinite(tab_mm) or tab_mm < 0:
        raise ValueError("Use positive unfold attempts and a finite, nonnegative tab width.")
    faces = box.faces
    ids = [{vertex: index for index, vertex in enumerate(face)} for face in faces]
    local = [panel_coordinates(box.points[list(face)]) for face in faces]
    incidence = {}
    for index, face in enumerate(faces):
        for a, b in face_edges(face):
            incidence.setdefault(edge_key(a, b), []).append((index, a, b))
    if any(len(items) != 2 for items in incidence.values()):
        raise InvalidRingBox("The surface does not have exactly two faces at each edge.")
    adjacency = [[] for _ in faces]
    for edge, ((left, _, _), (right, _, _)) in incidence.items():
        adjacency[left].append((right, edge))
        adjacency[right].append((left, edge))

    def checked_result(panels, hinges, preferred_tabs=None):
        tabs = make_tabs(panels, incidence, hinges, ids, tab_mm, rng, preferred_tabs) if tab_mm else {}
        if tabs is None:
            return None
        surface_area = sum(area2(p.xy) for p in panels.values())
        surface_area += sum(area2(tab.panel.xy) for tab in tabs.values())
        mass = surface_area * BOARD_THICKNESS_MM * BOARD_DENSITY_KG_M3 * 1e-6
        return Net(panels, frozenset(hinges), tabs, mass)

    if preferred is not None:
        hinges, preferred_tabs = preferred
        panels = {0: Panel.from_xy(local[0])}
        frontier = [0]
        while frontier:
            parent = frontier.pop()
            for child, edge in adjacency[parent]:
                if edge not in hinges or child in panels:
                    continue
                parent_xy = panels[parent].xy[[ids[parent][vertex] for vertex in edge]]
                candidate = align_panel(local, child, parent_xy, ids, edge)
                if not any(overlaps(candidate, panel) for panel in panels.values()):
                    panels[child] = candidate
                    frontier.append(child)
        if len(panels) == len(faces) and len(hinges) == len(faces) - 1:
            result = checked_result(panels, hinges, preferred_tabs)
            if result is not None:
                return result

    if len(box.bands) >= 3:
        strips = unfold_strips(box, local, ids)
        if strips is not None:
            for _ in range(3):
                result = checked_result(*strips)
                if result is not None:
                    return result

    for _ in range(attempts):
        panels = {0: Panel.from_xy(local[0])}
        hinges = set()
        frontier = [(0, child, edge) for child, edge in adjacency[0]]
        while frontier:
            i = rng.integers(len(frontier))
            parent, child, edge = frontier.pop(i)
            if child in panels:
                continue
            parent_xy = panels[parent].xy[[ids[parent][vertex] for vertex in edge]]
            candidate = align_panel(local, child, parent_xy, ids, edge)
            if any(overlaps(candidate, panel) for panel in panels.values()):
                continue
            panels[child] = candidate
            hinges.add(edge)
            frontier.extend((child, next_face, next_edge) for next_face, next_edge in adjacency[child]
                            if next_face not in panels)
        if len(panels) != len(faces):
            continue
        result = checked_result(panels, hinges)
        if result is not None:
            return result
    return None


def sample_ring_box(rng, *, sides=None, band_count=None):
    """Draw centered regular polygons with bounded radii, heights and twist."""
    sides = sides if sides is not None else int(rng.integers(4, 11))
    band_count = band_count if band_count is not None else int(rng.integers(3, 8))
    min_height = max(30, 9.5 * band_count + 3)
    height = rng.uniform(min_height, max(67, min_height + 22))
    spacings = rng.uniform(0.65, 1.35, band_count)
    if band_count >= 3:
        spacings = 8 + (height - 8 * band_count) * spacings / spacings.sum()
    else:
        spacings = height * spacings / spacings.sum()
    heights = np.r_[0, np.cumsum(spacings)]
    base = (PIZZA_RADIUS_MM + 1) / np.cos(np.pi / sides) + rng.uniform(3, 20)
    radii = base + np.r_[0, min(1, 4 / band_count) * rng.uniform(-5, 17, band_count)]
    step = 360 / sides
    twist_range = min(0.7, 1.7 / band_count)
    rotations = [0.0]
    bands = []
    for _ in range(band_count):
        target = rotations[-1] + rng.uniform(-twist_range, twist_range) * step
        phase = target % step
        shift = round((target - phase) / step) % sides
        rotations.append(phase)
        bands.append(Band(shift, bool(rng.integers(2))))
    return build_ring_box(sides, [Ring(*item) for item in zip(radii, heights, rotations)], bands)


def mutate_ring_box(box, rng):
    """Mutate a ring or band while preserving the case's rotational pattern."""
    sides = box.sides
    step = 360 / sides
    radii = [ring.radius_mm for ring in box.rings]
    heights = [ring.height_mm for ring in box.rings]
    twist = [((upper.angle_deg - lower.angle_deg + band.shift * step + 180) % 360) - 180
             for lower, upper, band in zip(box.rings, box.rings[1:], box.bands)]
    diagonals = [band.flip_diagonal for band in box.bands]
    move = rng.random()
    if move < 0.35:
        i = rng.integers(len(radii))
        radii[i] += rng.choice([-1, 1]) * rng.uniform(1.5, 6)
    elif move < 0.55:
        i = rng.integers(1, len(heights))
        upper = heights[i + 1] - 6 if i + 1 < len(heights) else 110
        heights[i] = float(np.clip(heights[i] + rng.choice([-1, 1]) * rng.uniform(2, 6),
                                   heights[i - 1] + 6, upper))
    elif move < 0.75:
        i = rng.integers(len(twist))
        twist[i] = float(np.clip(twist[i] + rng.choice([-1, 1]) * rng.uniform(0.02, 0.12) * step,
                                -0.7 * step, 0.7 * step))
    elif move < 0.90:
        i = rng.integers(len(diagonals))
        diagonals[i] = not diagonals[i]
    else:
        change = rng.choice(("insert", "delete", "sides"))
        if change == "insert" and len(twist) < 7:
            i = int(rng.integers(len(twist)))
            if heights[i + 1] - heights[i] < 12:
                raise InvalidRingBox("There is no room to insert this ring.")
            heights.insert(i + 1, (heights[i] + heights[i + 1]) / 2)
            radii.insert(i + 1, (radii[i] + radii[i + 1]) / 2 + rng.uniform(-3, 3))
            twist[i:i+1] = [twist[i] / 2, twist[i] / 2]
            diagonals.insert(i, diagonals[i])
        elif change == "delete" and len(twist) > 3:
            i = int(rng.integers(1, len(heights) - 1))
            heights.pop(i)
            radii.pop(i)
            twist[i-1:i+1] = [twist[i - 1] + twist[i]]
            diagonals.pop(i)
        else:
            new_sides = min(12, max(4, sides + rng.choice([-1, 1])))
            radii = [r * np.cos(np.pi / sides) / np.cos(np.pi / new_sides) for r in radii]
            twist = [d * sides / new_sides for d in twist]
            sides = int(new_sides)
            step = 360 / sides

    rotations = [0.0]
    bands = []
    for delta, direction in zip(twist, diagonals):
        target = rotations[-1] + float(delta)
        phase = target % step
        bands.append(Band(round((target - phase) / step) % sides, direction))
        rotations.append(phase)
    return build_ring_box(sides, [Ring(*item) for item in zip(radii, heights, rotations)], bands)


def write_obj(box, path):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("# Thin HeatPro ring-box cavity boundary; coordinates in mm\n")
        for x, y, z in box.points:
            stream.write(f"v {x:.10g} {y:.10g} {z:.10g}\n")
        for face in box.faces:
            stream.write("f " + " ".join(str(i + 1) for i in face) + "\n")


def write_net(box, net, path):
    """Write a 1:1 SVG. Black: cut. Dashed blue: fold."""
    sheets = [*net.panels.values(), *(tab.panel for tab in net.tabs.values())]
    points = np.vstack([panel.xy for panel in sheets])
    low = points.min(axis=0) - 5
    width, height = points.max(axis=0) - low + 5
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.5f}mm" height="{height:.5f}mm" '
             f'viewBox="0 0 {width:.5f} {height:.5f}">',
             '<title>HeatPro pattern: solid black cut lines and dashed blue fold lines</title>']

    def draw(a, b, fold=False):
        x1, y1 = a - low
        x2, y2 = b - low
        style = 'stroke="#136d9d" stroke-dasharray="2 1.5"' if fold else 'stroke="#202020"'
        lines.append(f'<line x1="{x1:.8f}" y1="{y1:.8f}" x2="{x2:.8f}" y2="{y2:.8f}" '
                     f'{style} stroke-width="0.25"/>')

    drawn = set()
    for i, face in enumerate(box.faces):
        for a, b in face_edges(face):
            edge = edge_key(a, b)
            start, end = net.panels[i].xy[[face.index(a), face.index(b)]]
            if edge in net.hinges:
                if edge not in drawn:
                    draw(start, end, fold=True)
                    drawn.add(edge)
            elif edge in net.tabs and net.tabs[edge].face == i:
                p, q, r, s = net.tabs[edge].panel.xy
                for left, right in ((start, p), (p, s), (s, r), (r, q), (q, end)):
                    draw(left, right)
                draw(p, q, fold=True)
            else:
                draw(start, end)
    lines.append("</svg>")
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--sides", type=int)
    parser.add_argument("--bands", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", type=Path, default=Path("ring_candidates"))
    parser.add_argument("--tab-mm", type=float, default=4.0)
    parser.add_argument("--nonconvex", action="store_true", help="Keep only genuinely indented cavities.")
    parser.add_argument("--unfold-attempts", type=int, default=32)
    parser.add_argument("--max-draws", type=int, default=200)
    args = parser.parse_args()
    if not (args.count > 0 and args.max_draws > 0 and args.unfold_attempts > 0
            and np.isfinite(args.tab_mm) and args.tab_mm >= 0 and (args.seed is None or args.seed >= 0)
            and (args.sides is None or 3 <= args.sides <= 24)
            and (args.bands is None or 1 <= args.bands <= 10)):
        parser.error("Use 3–24 sides, 1–10 bands, positive counts and a nonnegative tab width and seed.")
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    saved, serial = 0, 1
    for _ in range(args.max_draws):
        try:
            box = sample_ring_box(rng, sides=args.sides, band_count=args.bands)
        except InvalidRingBox:
            continue
        if args.nonconvex and box.convex_deficit < 1e-4:
            continue
        net = unfold(box, rng, attempts=args.unfold_attempts, tab_mm=args.tab_mm)
        if net is None or not 83 <= net.mass_g <= 249:
            continue
        while (args.out_dir / f"ring_{serial:03d}.svg").exists() or (args.out_dir / f"ring_{serial:03d}.obj").exists():
            serial += 1
        stem = args.out_dir / f"ring_{serial:03d}"
        write_net(box, net, stem.with_suffix(".svg"))
        write_obj(box, stem.with_suffix(".obj"))
        print(f"{stem.name} | {box.sides} sides, {len(box.bands)} bands, {len(box.faces)} faces, "
              f"{len(net.tabs)} tabs, {net.mass_g:.2f} g, {100 * box.convex_deficit:.2f}% convex-hull deficit",
              flush=True)
        saved += 1
        if saved == args.count:
            return
    parser.exit(1, f"Saved {saved}/{args.count} candidates in {args.max_draws} geometry draws.\n")


if __name__ == "__main__":
    main()
