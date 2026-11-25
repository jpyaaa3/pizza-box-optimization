"""Transient 3D board conduction coupled to lumped pizza and air temperatures."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


@dataclass
class ThermalMesh:
    xyz_m: np.ndarray
    tets: np.ndarray
    outer_floor: np.ndarray
    outer_air: np.ndarray
    pizza_contact: np.ndarray
    inner_air: np.ndarray
    air_volume_m3: float
    tet_xyz_m: np.ndarray | None = None


@dataclass
class ThermalResult:
    time_s: np.ndarray
    pizza_c: np.ndarray
    air_c: np.ndarray
    wall_sample_s: np.ndarray
    wall_face_c: np.ndarray
    wall_face_area_m2: np.ndarray
    ground_heat_j: float
    ambient_heat_j: float
    peak_balance_residual_j: float
    contact_area_m2: float
    board_volume_m3: float
    assembly_s: float
    factor_s: float
    integration_s: float


def surface_matrices(xyz, triangles):
    points = xyz[triangles]
    area = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    integral = np.bincount(triangles.ravel(), weights=np.repeat(area / 3, 3), minlength=len(xyz))
    mass = element_matrix(triangles, area[:, None, None] / 12 * (1 + np.eye(3)), len(xyz))
    return area, integral, mass


def element_matrix(elements, values, size):
    nodes = elements.shape[1]
    rows = np.repeat(elements, nodes, axis=1).ravel()
    cols = np.tile(elements, (1, nodes)).ravel()
    return coo_matrix((values.ravel(), (rows, cols)), shape=(size, size)).tocsr()


def solve_thermal(mesh: ThermalMesh, *, dt_s=1.0, duration_s=1200, wall_sample_s=10.0):
    """Solve in temperature rise over 22 °C using implicit Euler and linear tetrahedra."""
    if not np.all(np.isfinite([dt_s, duration_s, wall_sample_s])) or min(dt_s, duration_s, wall_sample_s) <= 0:
        raise ValueError("Time settings must be finite and positive.")
    steps = round(duration_s / dt_s)
    wall_stride = max(1, round(wall_sample_s / dt_s))
    if steps < 1 or not np.isclose(steps * dt_s, duration_s):
        raise ValueError("Duration must be a multiple of the time step.")

    start = perf_counter()
    xyz, tets = mesh.xyz_m, mesh.tets
    points = xyz[tets] if mesh.tet_xyz_m is None else mesh.tet_xyz_m
    edges = points[:, 1:] - points[:, :1]
    volumes = abs(np.linalg.det(edges)) / 6
    if np.any(volumes <= 0) or mesh.air_volume_m3 <= 0:
        raise ValueError("Positive tetrahedral and air volumes are required.")
    gradients = np.linalg.inv(edges).transpose(0, 2, 1)
    gradients = np.concatenate((-gradients.sum(axis=1, keepdims=True), gradients), axis=1)
    k_elements = 0.08 * volumes[:, None, None] * (gradients @ gradients.transpose(0, 2, 1))
    m_elements = (200 * 1400 / 20) * volumes[:, None, None] * (1 + np.eye(4))
    stiff = element_matrix(tets, k_elements, len(xyz))
    mass = element_matrix(tets, m_elements, len(xyz))
    del points, edges, gradients, k_elements, m_elements

    contact_area, contact_load, contact_mass = surface_matrices(xyz, mesh.pizza_contact)
    wall_area, wall_load, wall_mass = surface_matrices(xyz, mesh.inner_air)
    _, outside_load, outside_mass = surface_matrices(xyz, mesh.outer_air)
    contact_h, inside_h, outside_h = 10.0, 3.0, 10.0
    stiff += contact_h * contact_mass + inside_h * wall_mass + outside_h * outside_mass
    coupling = np.column_stack((contact_h * contact_load, inside_h * wall_load))
    pizza_area = np.pi * (0.15**2 + 0.30 * 0.015)
    pizza_air = inside_h * pizza_area
    capacity = np.array([0.750 * 3000, 1.2 * 1005 * mesh.air_volume_m3])
    lumps = np.array([
        [capacity[0] / dt_s + coupling[:, 0].sum() + pizza_air, -pizza_air],
        [-pizza_air, capacity[1] / dt_s + coupling[:, 1].sum() + pizza_air],
    ])

    fixed = np.unique(mesh.outer_floor)
    free = np.setdiff1d(np.arange(len(xyz)), fixed)
    m_free = mass[free][:, free] / dt_s
    system = (m_free + stiff[free][:, free]).tocsc()
    b_free = coupling[free]
    ground_mass = np.asarray(mass[fixed][:, free].sum(axis=0)).ravel() / dt_s
    ground_stiff = np.asarray(stiff[fixed][:, free].sum(axis=0)).ravel()
    ground_coupling = coupling[fixed].sum(axis=0)
    board_capacity = np.asarray(mass.sum(axis=0)).ravel()[free]
    outside = outside_h * outside_load[free]
    assembly_s = perf_counter() - start

    start = perf_counter()
    factor = splu(system)
    responses = factor.solve(b_free)
    lump_inverse = np.linalg.inv(lumps - b_free.T @ responses)
    factor_s = perf_counter() - start

    start = perf_counter()
    board = np.zeros(len(free))
    lump = np.array([68.0, 0.0])
    history = np.empty((steps + 1, 2))
    history[0] = lump
    sample_steps = np.unique(np.r_[np.arange(0, steps + 1, wall_stride), steps])
    samples = np.zeros((len(sample_steps), len(mesh.inner_air)), dtype=np.float32)
    global_to_free = np.full(len(xyz), len(free), dtype=int)
    global_to_free[free] = np.arange(len(free))
    wall_nodes = global_to_free[mesh.inner_air]
    sample_index = 1
    energy = float(capacity @ lump)
    ground_heat = ambient_heat = residual = 0.0
    for step in range(1, steps + 1):
        forced_board = factor.solve(m_free @ board)
        lump = lump_inverse @ (capacity / dt_s * lump + np.einsum('ij,i->j', b_free, forced_board))
        updated = forced_board + responses[:, 0] * lump[0] + responses[:, 1] * lump[1]
        ground = -np.einsum('i,i->', ground_mass, updated - board) - np.einsum('i,i->', ground_stiff, updated)
        ground += ground_coupling @ lump
        ambient = np.einsum('i,i->', outside, updated)
        new_energy = np.einsum('i,i->', board_capacity, updated) + capacity @ lump
        residual = max(residual, abs(new_energy - energy + dt_s * (ground + ambient)))
        ground_heat += dt_s * ground
        ambient_heat += dt_s * ambient
        energy = new_energy
        board = updated
        history[step] = lump
        if sample_index < len(sample_steps) and step == sample_steps[sample_index]:
            samples[sample_index] = np.r_[board, 0.0][wall_nodes].mean(axis=1)
            sample_index += 1

    return ThermalResult(
        np.arange(steps + 1) * dt_s, history[:, 0] + 22, history[:, 1] + 22,
        sample_steps * dt_s, samples + 22, wall_area, ground_heat, ambient_heat, residual,
        float(contact_area.sum()), float(volumes.sum()), assembly_s, factor_s, perf_counter() - start,
    )
