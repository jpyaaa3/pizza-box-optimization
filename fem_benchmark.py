"""Mesh the HeatPro shell in Gmsh and time a 20-minute thermal FEM solve."""

import argparse
from collections import defaultdict
from time import perf_counter

import gmsh
import numpy as np

from box_generator import BoxDesign, build_box, sample_box
from thermal_fem import ThermalMesh, solve_thermal


class MeshingError(RuntimeError):
    """The generator geometry could not be meshed."""


def polygon_faces(hull):
    groups = defaultdict(set)
    for nodes, equation in zip(hull.simplices, hull.equations):
        groups[tuple(np.round(equation, 8))].update(nodes)
    for equation, group in groups.items():
        ids = np.array(sorted(group))
        normal = np.array(equation[:3])
        positions = hull.points[ids]
        offsets = positions - positions.mean(axis=0)
        x_axis = offsets[0] / np.linalg.norm(offsets[0])
        y_axis = np.cross(normal, x_axis)
        yield ids[np.argsort(np.arctan2(offsets @ y_axis, offsets @ x_axis))]


def add_hull(hull, floor_z, *, imprint_pizza=False):
    occ = gmsh.model.occ
    points = {int(i): occ.addPoint(*hull.points[i]) for i in hull.vertices}
    edges = {}
    surfaces, floor, contact = [], [], []
    faces = hull.polygons if hasattr(hull, "polygons") else polygon_faces(hull)
    for polygon in faces:
        lines = []
        for a, b in zip(polygon, np.roll(polygon, -1)):
            start, end = sorted((a, b))
            edge = (start, end)
            if edge not in edges:
                edges[edge] = occ.addLine(points[start], points[end])
            lines.append(edges[edge] if a == start else -edges[edge])
        loop = occ.addCurveLoop(lines)
        is_floor = np.allclose(hull.points[polygon, 2], floor_z, atol=1e-6, rtol=0)
        if is_floor and imprint_pizza:
            circle = occ.addCircle(0, 0, floor_z, 150)
            disk = occ.addCurveLoop([circle])
            surface = occ.addPlaneSurface([loop, disk])
            contact.append(occ.addPlaneSurface([disk]))
        else:
            surface = occ.addPlaneSurface([loop])
        if is_floor:
            floor.append(surface)
        surfaces.append(surface)
    if len(floor) != 1 or (imprint_pizza and len(contact) != 1):
        raise ValueError("Expected exactly one planar floor and one pizza contact disk.")
    return occ.addSurfaceLoop(surfaces + contact), surfaces, floor, contact


def mesh_box(box, size_mm):
    """Mesh the solid between its two boundaries. Build in mm and return m."""
    gmsh.clear()
    gmsh.model.add("pizza_box")
    outer_loop, outer, ground, _ = add_hull(box.outer, -3)
    inner_loop, inner, _, disk = add_hull(box.cavity, 0, imprint_pizza=True)
    volume = gmsh.model.occ.addVolume([outer_loop, inner_loop])
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
    gmsh.model.mesh.generate(3)

    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    tags = tags[order]

    def elements(kind, entities, width):
        blocks = [gmsh.model.mesh.getElementsByType(kind, tag)[1] for tag in entities]
        node_tags = np.concatenate(blocks).reshape(-1, width)
        if not node_tags.size:
            raise ValueError("An expected FEM region has no elements.")
        return np.searchsorted(tags, node_tags)

    return ThermalMesh(
        coordinates.reshape(-1, 3)[order] * 1e-3,
        elements(4, [volume], 4), elements(2, ground, 3),
        elements(2, [tag for tag in outer if tag not in ground], 3),
        elements(2, disk, 3), elements(2, inner, 3), box.air_volume_m3,
    )


def reference_box():
    design = BoxDesign((0, 90, 180, 270), (160, 160, 160, 160),
                       ((-160, -160, 50), (-160, 160, 50), (160, -160, 50), (160, 160, 50)))
    return build_box(design)


def candidates(seed, count):
    rng = np.random.default_rng(seed)
    pool = [sample_box(rng) for _ in range(3 * count)]
    pool = sorted((box for box in pool if 83 <= box.mass_g <= 249), key=lambda box: box.air_volume_m3)
    while len(pool) < count:
        box = sample_box(rng)
        if 83 <= box.mass_g <= 249:
            pool.append(box)
    pool.sort(key=lambda box: box.air_volume_m3)
    return [pool[i] for i in np.linspace(0, len(pool) - 1, count, dtype=int)]


def run_case(name, box, size_mm, dt_s):
    started = perf_counter()
    try:
        mesh = mesh_box(box, size_mm)
    except Exception as error:
        # The Gmsh Python API reports CAD and meshing failures as plain Exception.
        raise MeshingError(str(error)) from error
    mesh_s = perf_counter() - started
    result = solve_thermal(mesh, dt_s=dt_s)
    wall_c = np.average(result.wall_face_c[-1], weights=result.wall_face_area_m2)
    volume_error = 100 * (result.board_volume_m3 * 200_000 / box.mass_g - 1)
    area_error = 100 * (result.contact_area_m2 / box.contact_area_m2 - 1)
    print(f"{name}: h={size_mm:g} mm, dt={dt_s:g} s, nodes={len(mesh.xyz_m)}, tets={len(mesh.tets)}, "
          f"pizza={result.pizza_c[-1]:.5f} C, air={result.air_c[-1]:.5f} C, wall_mean={wall_c:.5f} C", flush=True)
    print(f"  mesh={mesh_s:.3f} s, assembly={result.assembly_s:.3f} s, LU={result.factor_s:.3f} s, "
          f"time_steps={result.integration_s:.3f} s, total={perf_counter() - started:.3f} s", flush=True)
    print(f"  contact_area_error={area_error:+.4f}%, board_volume_error={volume_error:+.4f}%, "
          f"ground_heat={result.ground_heat_j:.1f} J, ambient_heat={result.ambient_heat_j:.1f} J, "
          f"peak_balance_residual={result.peak_balance_residual_j:.2e} J", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-mm", type=float, default=4.0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not np.isfinite(args.size_mm) or args.size_mm <= 0 or args.candidates < 0:
        parser.error("Use a positive mesh size and a nonnegative candidate count.")
    gmsh.initialize([], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    failures = 0
    try:
        run_case("reference", reference_box(), args.size_mm, args.dt_s)
        for index, box in enumerate(candidates(args.seed, args.candidates) if args.candidates else [], 1):
            try:
                run_case(f"candidate_{index:02d}", box, args.size_mm, args.dt_s)
            except (RuntimeError, ValueError) as error:
                failures += 1
                print(f"candidate_{index:02d} failed: {error}", flush=True)
    finally:
        gmsh.finalize()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
