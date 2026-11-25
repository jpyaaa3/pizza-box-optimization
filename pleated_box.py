"""Generate nonconvex, flat-bottom pizza boxes with irregular vertical pleats."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from box_generator import (
    BOARD_DENSITY_KG_M3,
    BOARD_THICKNESS_MM,
    InvalidGeometry,
    PIZZA_HEIGHT_MM,
    PIZZA_RADIUS_MM,
)


@dataclass(frozen=True)
class Boundary:
    points: np.ndarray
    polygons: tuple[np.ndarray, ...]

    @property
    def vertices(self):
        return range(len(self.points))


@dataclass(frozen=True)
class PleatedBox:
    cavity: Boundary
    outer: Boundary
    height_mm: float
    floor_area_m2: float
    side_area_m2: float
    mass_g: float
    side_area_ratio: float

    @property
    def contact_area_m2(self):
        return np.pi * (PIZZA_RADIUS_MM / 1000) ** 2

    @property
    def air_volume_m3(self):
        return (self.floor_area_m2 * self.height_mm - self.contact_area_m2 * PIZZA_HEIGHT_MM) / 1000

    @property
    def nonfloor_area_m2(self):
        return self.floor_area_m2 + self.side_area_m2


def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def check_ring(ring):
    """One counterclockwise revolution with short angular steps is a simple ring."""
    angles = np.arctan2(ring[:, 1], ring[:, 0])
    steps = (np.roll(angles, -1) - angles) % (2 * np.pi)
    if np.any(steps < 1e-8) or np.any(steps >= np.pi) or not np.isclose(steps.sum(), 2 * np.pi):
        raise InvalidGeometry("The side outline must make one counterclockwise turn without crossing itself.")


def area(ring):
    return 0.5 * cross(ring, np.roll(ring, -1, axis=0)).sum()


def segment_intersections(inner, outer):
    a = inner[:, None, :]
    b = np.roll(inner, -1, axis=0)[:, None, :]
    c = outer[None, :, :]
    d = np.roll(outer, -1, axis=0)[None, :, :]
    separate = np.any(np.maximum(a, b) < np.minimum(c, d) - 1e-8, axis=2)
    separate |= np.any(np.maximum(c, d) < np.minimum(a, b) - 1e-8, axis=2)
    signs_ab = np.stack((cross(b - a, c - a), cross(b - a, d - a)))
    signs_cd = np.stack((cross(d - c, a - c), cross(d - c, b - c)))
    for values in (signs_ab, signs_cd):
        separate |= (values > 1e-8).all(axis=0) | (values < -1e-8).all(axis=0)
    return ~separate


def prism(ring, bottom, top):
    count = len(ring)
    indexes = np.arange(count)
    next_indexes = np.roll(indexes, -1)
    points = np.column_stack((np.tile(ring, (2, 1)), np.repeat([bottom, top], count)))
    sides = np.column_stack((indexes, next_indexes, next_indexes + count, indexes + count))
    return Boundary(points, (indexes[::-1], indexes + count, *sides))


def build_pleated_box(outline_mm, height_mm, *, clearance_mm=1.0):
    """Form a vertical shell from a star-shaped, possibly concave inner outline."""
    inner = np.asarray(outline_mm, dtype=float)
    if inner.ndim != 2 or inner.shape[1] != 2 or len(inner) < 3 or not np.isfinite(inner).all():
        raise InvalidGeometry("Provide at least three finite (x, y) outline points.")
    if not np.isfinite(clearance_mm) or clearance_mm < 0:
        raise InvalidGeometry("Clearance must be finite and nonnegative.")
    if not np.isfinite(height_mm) or height_mm < PIZZA_HEIGHT_MM + clearance_mm:
        raise InvalidGeometry("The lid does not clear the pizza.")
    check_ring(inner)
    edges = np.roll(inner, -1, axis=0) - inner
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    offsets = np.einsum("ij,ij->i", normals, inner)
    along_edge = np.clip(-np.einsum("ij,ij->i", inner, edges) / lengths**2, 0, 1)
    closest = inner + along_edge[:, None] * edges
    if np.any(np.linalg.norm(closest, axis=1) < PIZZA_RADIUS_MM + clearance_mm - 1e-7):
        raise InvalidGeometry("A wall cuts into the pizza or its clearance.")
    thick_offsets = offsets + BOARD_THICKNESS_MM
    systems = np.stack((np.roll(normals, 1, axis=0), normals), axis=1)
    if np.any(np.abs(np.linalg.det(systems)) < 1e-8):
        raise InvalidGeometry("Adjacent side panels are almost collinear, making the miter ambiguous.")
    rhs = np.column_stack((np.roll(thick_offsets, 1), thick_offsets))
    outer = np.linalg.solve(systems, rhs[..., None])[..., 0]
    outer_edges = np.roll(outer, -1, axis=0) - outer
    collisions = np.triu(segment_intersections(outer, outer), 2)
    collisions[0, -1] = False  # The first and last edges share a vertex.
    if (np.any(np.einsum("ij,ij->i", outer_edges, edges) <= 0) or collisions.any()
            or segment_intersections(inner, outer).any()):
        raise InvalidGeometry("The 3 mm offset closes or crosses a fold.")
    inner_area, outer_area = area(inner), area(outer)
    next_outer = outer + outer_edges
    winding = np.arctan2(cross(outer, next_outer), np.einsum("ij,ij->i", outer, next_outer)).sum()
    if outer_area <= inner_area or not np.isclose(winding, 2 * np.pi):
        raise InvalidGeometry("The offset does not enclose the cavity.")
    perimeter = lengths.sum()
    volume_mm3 = outer_area * (height_mm + 2 * BOARD_THICKNESS_MM) - inner_area * height_mm
    return PleatedBox(
        prism(inner, 0, height_mm), prism(outer, -BOARD_THICKNESS_MM, height_mm + BOARD_THICKNESS_MM),
        height_mm, inner_area * 1e-6, perimeter * height_mm * 1e-6,
        volume_mm3 * BOARD_DENSITY_KG_M3 * 1e-6, perimeter / ConvexHull(inner).area,
    )


def sample_pleated_box(rng, *, folds=None, depth_mm=None, attempts=400):
    """Draw irregular ridge and valley spacing, depths, clearance, and box height."""
    if folds is not None and folds < 3:
        raise ValueError("Use at least three folds.")
    if depth_mm is not None and (not np.isfinite(depth_mm) or depth_mm < 0):
        raise ValueError("Use a finite, nonnegative pleat depth.")
    for _ in range(attempts):
        count = folds if folds is not None else int(rng.integers(5, 25))
        depth = depth_mm if depth_mm is not None else rng.uniform(4, 24)
        steps = rng.uniform(0.7, 1.3, 2 * count)
        steps *= 2 * np.pi / steps.sum()
        angles = rng.uniform(0, 2 * np.pi) + np.r_[0, np.cumsum(steps[:-1])]
        radius = (PIZZA_RADIUS_MM + 1) / np.cos(steps.max() / 2) + rng.uniform(1, 7)
        radii = np.full(len(angles), radius)
        radii[::2] += rng.uniform(0.75, 1.25, count) * depth
        ring = radii[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
        try:
            box = build_pleated_box(ring, rng.uniform(20, 52))
        except InvalidGeometry:
            continue
        if 83 <= box.mass_g <= 249:
            return box
    raise InvalidGeometry(f"No valid pleated box found in {attempts} draws.")


def write_pleated_obj(box, path):
    """Write the shell boundary, with the outside floor at z=0."""
    offset = 1
    with Path(path).open("w", encoding="utf-8") as stream:
        stream.write("# HeatPro pleated shell, units: mm\n")
        for name, boundary in (("outer", box.outer), ("cavity", box.cavity)):
            stream.write(f"g {name}\n")
            for x, y, z in boundary.points:
                stream.write(f"v {x:.12g} {y:.12g} {z + BOARD_THICKNESS_MM:.12g}\n")
            for face in boundary.polygons:
                indexes = face if name == "outer" else face[::-1]
                stream.write("f " + " ".join(str(i + offset) for i in indexes) + "\n")
            offset += len(boundary.points)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--depth-mm", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--obj-dir", type=Path)
    args = parser.parse_args()
    if args.count < 1 or (args.seed is not None and args.seed < 0):
        parser.error("Use a positive count and a nonnegative seed.")
    rng = np.random.default_rng(args.seed)
    if args.obj_dir is not None:
        args.obj_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.count + 1):
        try:
            box = sample_pleated_box(rng, folds=args.folds, depth_mm=args.depth_mm)
        except ValueError as error:
            parser.error(str(error))
        count = len(box.cavity.points) // 4
        print(f"{index}: {count} folds, {box.mass_g:.2f} g, side area {box.side_area_m2:.4f} m^2, "
              f"{(box.side_area_ratio - 1) * 100:+.1f}% vs. its convex outline")
        if args.obj_dir is not None:
            write_pleated_obj(box, args.obj_dir / f"pleated_{index:03d}.obj")


if __name__ == "__main__":
    main()
