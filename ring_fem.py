"""Mesh folded ring panels as 3D layers with shared crease temperatures."""

import gmsh
import numpy as np

from box_generator import BOARD_THICKNESS_MM
from fem_benchmark import add_hull
from pleated_box import Boundary
from thermal_fem import ThermalMesh


def unit_normal(points):
    direction = np.cross(points[1] - points[0], points[2] - points[0])
    return direction / np.linalg.norm(direction)


def mesh_ring_box(box, size_mm, *, layers=2):
    """Extrude each surface element normally, identifying nodes at shared creases.

    Each panel has its own 3D coordinates and exactly 3 mm thickness. The
    coincident DOF numbering across panel edges is an idealized folded joint;
    no junction overlap, extra tab conduction, or CAD wall offset is assumed.
    """
    if layers < 1:
        raise ValueError("At least one board layer is needed.")
    gmsh.clear()
    gmsh.model.add("layered_ring_box")
    boundary = Boundary(box.points, tuple(np.asarray(face) for face in box.faces))
    _, walls, floor, disk = add_hull(boundary, 0, imprint_pizza=True)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", size_mm)
    gmsh.model.mesh.generate(2)

    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    order = np.argsort(tags)
    coordinates = coordinates.reshape(-1, 3)[order] * 1e-3
    tags = tags[order]
    entity_tags = [*walls, *disk]
    tri_blocks = []
    directions = []
    for index, entity in enumerate(entity_tags):
        values = gmsh.model.mesh.getElementsByType(2, entity)[1].reshape(-1, 3)
        if not len(values):
            raise ValueError(f"Ring surface {index} has no mesh triangles.")
        tri_blocks.append(np.sort(np.searchsorted(tags, values), axis=1))
        face = box.faces[index] if index < len(walls) else box.faces[0]
        directions.append(np.repeat(unit_normal(box.points[list(face)])[None, :], len(values), axis=0))
    directions = np.concatenate(directions)
    triangles = np.concatenate(tri_blocks)
    used, local_indices = np.unique(triangles, return_inverse=True)
    triangles = local_indices.reshape(-1, 3)
    coordinates = coordinates[used]
    starts = np.r_[0, np.cumsum([len(block) for block in tri_blocks])]
    disk_start = starts[len(walls)]
    floor_index = walls.index(floor[0])
    ground = np.concatenate((triangles[starts[floor_index]:starts[floor_index+1]], triangles[disk_start:]))
    exposed = np.concatenate((triangles[:starts[floor_index]], triangles[starts[floor_index+1]:disk_start]))
    node_count = len(coordinates)

    # The common diagonal of each lateral prism face joins its lower smaller
    # node to its upper larger node, so neighbors share the same tetra faces.
    pieces = np.array([[0, 1, 2, 5], [0, 1, 4, 5], [0, 3, 4, 5]])
    ref = coordinates[triangles]
    rise = (BOARD_THICKNESS_MM * 1e-3 / layers) * directions[:, None, :]
    tets, tet_xyz = [], []
    for layer in range(layers):
        prism_nodes = np.column_stack((triangles + layer * node_count, triangles + (layer + 1) * node_count))
        prism_xyz = np.concatenate((ref + layer * rise, ref + (layer + 1) * rise), axis=1)
        tets.append(prism_nodes[:, pieces].reshape(-1, 4))
        tet_xyz.append(prism_xyz[:, pieces, :].reshape(-1, 4, 3))

    outer_start = layers * node_count
    return ThermalMesh(
        np.tile(coordinates, (layers + 1, 1)), np.concatenate(tets), ground + outer_start,
        exposed + outer_start, triangles[disk_start:], triangles[:disk_start], box.air_volume_m3,
        tet_xyz_m=np.concatenate(tet_xyz),
    )
