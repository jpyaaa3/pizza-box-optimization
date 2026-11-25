"""Generate convex pizza-box shells from a base polygon and upper points."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


PIZZA_RADIUS_MM = 150.0
PIZZA_HEIGHT_MM = 15.0
BOARD_THICKNESS_MM = 3.0
BOARD_DENSITY_KG_M3 = 200.0
TOLERANCE_MM = 1e-7


class InvalidGeometry(ValueError):
    """An input design cannot produce a valid shell around the pizza."""


@dataclass(frozen=True)
class BoxDesign:
    """Each base line is x*cos(angle) + y*sin(angle) = distance."""

    base_angles_deg: tuple[float, ...]
    base_distances_mm: tuple[float, ...]
    upper_points_mm: tuple[tuple[float, float, float], ...]


@dataclass(frozen=True)
class BoxGeometry:
    design: BoxDesign
    cavity: ConvexHull
    outer: ConvexHull
    floor_area_m2: float
    nonfloor_area_m2: float

    @property
    def contact_area_m2(self):
        return np.pi * (PIZZA_RADIUS_MM / 1000) ** 2

    @property
    def exposed_floor_area_m2(self):
        return max(0.0, self.floor_area_m2 - self.contact_area_m2)

    @property
    def cavity_volume_m3(self):
        return self.cavity.volume * 1e-9

    @property
    def air_volume_m3(self):
        return self.cavity_volume_m3 - self.contact_area_m2 * PIZZA_HEIGHT_MM / 1000

    @property
    def mass_g(self):
        return (self.outer.volume - self.cavity.volume) * 1e-6 * BOARD_DENSITY_KG_M3


def build_box(design: BoxDesign, *, clearance_mm: float = 0.0) -> BoxGeometry:
    """Build the shell in mm, with the pizza centered on the cavity floor at z=0.

    Clearance applies to every nonfloor face and is measured normal to it.
    Mass represents the ideal solid before fabrication tabs and overlaps.
    """
    angles = np.asarray(design.base_angles_deg, dtype=float)
    distances = np.asarray(design.base_distances_mm, dtype=float)
    upper = np.asarray(design.upper_points_mm, dtype=float)
    if angles.ndim != 1 or angles.size < 3 or distances.shape != angles.shape:
        raise InvalidGeometry("Provide at least three base angles and one distance per angle.")
    if upper.ndim != 2 or upper.shape[1] != 3 or len(upper) < 1:
        raise InvalidGeometry("Provide one or more upper points as (x, y, z) coordinates.")
    if not all(np.isfinite(values).all() for values in (angles, distances, upper)):
        raise InvalidGeometry("All coordinates, angles, and distances must be finite.")
    if not np.isfinite(clearance_mm) or clearance_mm < 0:
        raise InvalidGeometry("Clearance must be finite and nonnegative.")
    if np.any(upper[:, 2] <= TOLERANCE_MM):
        raise InvalidGeometry("Upper points must be above the cavity floor.")
    if np.any(distances < PIZZA_RADIUS_MM + clearance_mm - TOLERANCE_MM):
        raise InvalidGeometry("The base does not enclose the pizza with the requested clearance.")

    angles = np.deg2rad(angles % 360)
    sorted_angles = np.sort(angles)
    gaps = np.diff(np.r_[sorted_angles, sorted_angles[0] + 2 * np.pi])
    if np.any(gaps <= 1e-10) or np.any(gaps >= np.pi - 1e-10):
        raise InvalidGeometry("Base directions must be distinct and enclose a bounded polygon.")
    base_planes = np.column_stack((np.cos(angles), np.sin(angles), -distances))

    try:
        base = HalfspaceIntersection(base_planes, np.zeros(2)).intersections
        if len(base) != len(angles):
            raise InvalidGeometry("Each base line must contribute one edge to the polygon.")
        floor_points = np.column_stack((base, np.zeros(len(base))))
        cavity = ConvexHull(np.vstack((floor_points, upper)))

        # Qhull triangulates faces, so multiple equations can describe one plane.
        equations = cavity.equations / np.linalg.norm(cavity.equations[:, :3], axis=1)[:, None]
        normals, offsets = equations[:, :3], equations[:, 3]
        floor_faces = np.all(np.abs(cavity.points[cavity.simplices, 2]) <= TOLERANCE_MM, axis=1)
        pizza_support = (
            PIZZA_RADIUS_MM * np.linalg.norm(normals[:, :2], axis=1)
            + PIZZA_HEIGHT_MM * np.maximum(normals[:, 2], 0.0)
        )
        if np.any((pizza_support + offsets)[~floor_faces] > TOLERANCE_MM - clearance_mm):
            raise InvalidGeometry("A cavity face cuts into the pizza or its clearance.")

        _, first = np.unique(np.round(equations, decimals=9), axis=0, return_index=True)
        outer_planes = equations[first].copy()
        outer_planes[:, 3] -= BOARD_THICKNESS_MM
        inside = np.array([0.0, 0.0, PIZZA_HEIGHT_MM / 2])
        outer_points = HalfspaceIntersection(outer_planes, inside).intersections
        outer = ConvexHull(outer_points)
    except QhullError as error:
        raise InvalidGeometry("The points or planes do not define a numerically valid solid.") from error

    triangles = cavity.points[cavity.simplices[floor_faces]]
    floor_area_mm2 = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1
    ).sum()
    return BoxGeometry(
        design, cavity, outer, floor_area_mm2 * 1e-6, (cavity.area - floor_area_mm2) * 1e-6
    )


def sample_box(
    rng: np.random.Generator, *, base_sides: int | None = None,
    upper_count: int | None = None, clearance_mm: float = 1.0, attempts: int = 300,
) -> BoxGeometry:
    """Sample a random hull. Omitted counts are drawn again on each proposal."""
    if base_sides is not None and base_sides < 3:
        raise ValueError("Use at least three base sides.")
    if upper_count is not None and upper_count < 1:
        raise ValueError("Use at least one upper point.")
    if attempts < 1:
        raise ValueError("Use at least one attempt.")
    if not np.isfinite(clearance_mm) or clearance_mm < 0:
        raise ValueError("Clearance must be finite and nonnegative.")

    for _ in range(attempts):
        sides = base_sides if base_sides is not None else int(rng.integers(3, 13))
        points = upper_count if upper_count is not None else int(rng.integers(1, 17))
        base_angles = rng.uniform(0, 360, sides)
        base_distances = PIZZA_RADIUS_MM + clearance_mm + rng.uniform(3, 30, sides)
        extent = PIZZA_RADIUS_MM + clearance_mm + 60
        ceiling = PIZZA_HEIGHT_MM + clearance_mm
        upper = rng.uniform((-extent, -extent, ceiling + 2),
                            (extent, extent, ceiling + 85), size=(points, 3))
        design = BoxDesign(tuple(base_angles), tuple(base_distances), tuple(map(tuple, upper)))
        try:
            return build_box(design, clearance_mm=clearance_mm)
        except InvalidGeometry:
            continue
    raise InvalidGeometry(f"No valid candidate in {attempts} draws. Try more upper points or specify them directly.")


def write_obj(box: BoxGeometry, path: Path) -> None:
    """Write the shell boundary in mm, with the exterior bottom at z=0."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        output.write("# HeatPro shell, units: mm\n")
        offset = 1
        for name, hull, reverse in (("outer", box.outer, False), ("cavity", box.cavity, True)):
            output.write(f"g {name}\n")
            indices = dict(zip(hull.vertices, range(offset, offset + len(hull.vertices))))
            for x, y, z in hull.points[hull.vertices]:
                output.write(f"v {x:.12g} {y:.12g} {z + BOARD_THICKNESS_MM:.12g}\n")
            for triangle, equation in zip(hull.simplices, hull.equations):
                a, b, c = hull.points[triangle]
                inward = np.dot(np.cross(b - a, c - a), equation[:3]) < 0
                if inward != reverse:
                    triangle = triangle[::-1]
                output.write("f " + " ".join(str(indices[index]) for index in triangle) + "\n")
            offset += len(hull.vertices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-sides", type=int, help="Fix the base side count (default: random 3–12).")
    parser.add_argument("--upper-points", type=int, help="Fix the upper point count (default: random 1–16).")
    parser.add_argument("--clearance-mm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, help="Set a seed to reproduce the same candidate sequence.")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--obj-dir", type=Path)
    args = parser.parse_args()
    if args.count < 1 or (args.seed is not None and args.seed < 0):
        parser.error("Use a positive count and a nonnegative seed.")

    rng = np.random.default_rng(args.seed)
    if args.obj_dir is not None:
        args.obj_dir.mkdir(parents=True, exist_ok=True)
    for index in range(1, args.count + 1):
        try:
            box = sample_box(rng, base_sides=args.base_sides, upper_count=args.upper_points,
                             clearance_mm=args.clearance_mm)
        except ValueError as error:
            parser.error(str(error))
        mass_in_budget = "yes" if 83 <= box.mass_g <= 249 else "no"
        print(f"{index}: {len(box.design.base_angles_deg)} base sides, "
              f"{len(box.design.upper_points_mm)} upper points, "
              f"{box.mass_g:.2f} g, 83-249 g: {mass_in_budget}, "
              f"free air {1000 * box.air_volume_m3:.3f} L")
        if args.obj_dir is not None:
            write_obj(box, args.obj_dir / f"box_{index:03d}.obj")


if __name__ == "__main__":
    main()
