"""Make a flat, grooved board from a HeatPro ring_box OBJ and its original SVG.

Usage: python grooved_blank.py RUN_DIRECTORY
Writes flat.obj, flat.stp and folded.stp in each of rough/ and smooth/.
Units are mm. z=0 is the retained inner skin, z=thickness the machined side.
Each panel is a closed OBJ group. Do not weld coincident slit faces.
"""
import argparse
import itertools
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation


def cyclic(items):
    return zip(items, items[1:] + items[:1])


def read_surface(path):
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        words = line.split()
        if words[:1] == ["v"]:
            vertices.append(list(map(float, words[1:4])))
        if words[:1] == ["f"]:
            faces.append([int(x.split("/")[0]) - 1 for x in words[1:]])
    return np.array(vertices), faces


def read_layout(root, vertices, faces):
    # Consume draw calls in ring_box.write_net order. Tab folds are also dashed.
    lines = []
    for el in root.findall("{http://www.w3.org/2000/svg}line"):
        coords = [float(el.attrib[k]) for k in ("x1", "y1", "x2", "y2")]
        lines.append((np.array(coords).reshape(2, 2), el))
    hinges, panels, incidence = {}, [], {}
    cursor = 0
    for i, face in enumerate(faces):
        segments = []
        for a, b in cyclic(face):
            key = tuple(sorted((a, b)))
            incidence.setdefault(key, []).append((i, a, b))
            if key in hinges:
                segment = hinges[key][0][::-1]
            else:
                segment, el = lines[cursor]
                cursor += 1
                length = np.linalg.norm(vertices[b] - vertices[a])
                if "stroke-dasharray" in el.attrib:
                    hinges[key] = segment, el
                elif abs(np.linalg.norm(segment[1] - segment[0]) - length) > 1e-5:
                    block = lines[cursor-1:cursor+5]
                    if (len(block) != 6 or any("stroke-dasharray" in s.attrib for _, s in block[:5])
                            or "stroke-dasharray" not in block[5][1].attrib
                            or any(np.linalg.norm(block[k][0][1]-block[k+1][0][0]) > 1e-5 for k in range(4))):
                        raise ValueError("Unexpected tab lines in SVG")
                    tab = np.array([block[k][0][1] for k in range(4)])
                    if not np.allclose(tab[[0,3]], block[5][0], atol=1e-5, rtol=0):
                        raise ValueError("Tab fold does not match its outline")
                    segment = np.array([segment[0], block[4][0][1]])
                    cursor += 5
            segments.append(segment)
        xy = np.array([s[0] for s in segments])
        spatial = vertices[face]
        if (any(np.linalg.norm(s[1]-t[0]) > 1e-5 for s,t in cyclic(segments))
                or not np.allclose(np.linalg.norm(xy[:,None]-xy, axis=2),
                                   np.linalg.norm(spatial[:,None]-spatial, axis=2), atol=1e-5, rtol=0)):
            raise ValueError(f"SVG panel {i+1} does not match OBJ")
        panels.append(xy)
    if cursor != len(lines) or len(hinges) != len(faces)-1:
        raise ValueError("Expected an original, one-piece HeatPro ring-box net")
    return panels, hinges, incidence


def panel_mesh(xy, slopes, thickness):
    # Convex panel: inward edge distance >= z * tan(half groove angle).
    # Intersecting all half-spaces also trims grooves where several meet.
    planes = [[0,0,-1,0], [0,0,1,thickness]]
    for (a,b), slope in zip(cyclic(list(xy)), slopes):
        dx, dy = b-a
        n = np.array([-dy,dx])/math.hypot(dx,dy)
        if (xy.mean(axis=0)-a) @ n > 0:
            n = -n
        planes.append([*n, slope, n @ a])
    return clipped_panel(np.array(planes))


def clipped_panel(planes):
    corners = []
    for triple in itertools.combinations(planes, 3):
        system = np.array(triple)
        if abs(np.linalg.det(system[:,:3])) < 1e-10:
            continue
        point = np.linalg.solve(system[:,:3], system[:,3])
        if (planes[:,:3] @ point <= planes[:,3]+1e-7).all():
            corners.append(point)
    pts = np.unique(np.round(corners, 8), axis=0)
    hull = ConvexHull(pts)
    triangles = hull.simplices.copy()
    for tri, plane in zip(triangles, hull.equations):
        a,b,c = pts[tri]
        if np.cross(b-a,c-a) @ plane[:3] < 0:
            tri[1],tri[2] = tri[2],tri[1]
    return pts, triangles, planes


def penetrates(a, b, tolerance=1e-5):
    if np.min(np.minimum(a.max(0),b.max(0))-np.maximum(a.min(0),b.min(0))) <= tolerance:
        return False
    ha, hb = ConvexHull(a), ConvexHull(b)
    def edges(points, hull):
        triangles = points[hull.simplices]
        return (triangles-np.roll(triangles,1,axis=1)).reshape(-1,3)
    axes = np.vstack((ha.equations[:,:3], hb.equations[:,:3],
                      np.cross(edges(a,ha)[:,None],edges(b,hb)).reshape(-1,3)))
    lengths = np.linalg.norm(axes,axis=1)
    axes = axes[lengths>1e-8]/lengths[lengths>1e-8,None]
    pa,pb = a@axes.T,b@axes.T
    return np.all(np.minimum(pa.max(0),pb.max(0))-np.maximum(pa.min(0),pb.min(0)) > tolerance)


def relieve_corners(groups, poses, vertices, faces, panels):
    """Separate colliding corner pairs by a common plane preserving both inner faces."""
    def positioned(i):
        rotation, shift = poses[i]
        return groups[i][1] @ rotation.T + shift
    solids = [positioned(i) for i in range(len(groups))]
    count = 0
    for i,j in itertools.combinations(range(len(groups)),2):
        if not penetrates(solids[i],solids[j]):
            continue
        common = set(faces[i]) & set(faces[j])
        if len(common) != 1:
            raise ValueError(f"Folded panels {i+1} and {j+1} interfere beyond one corner")
        vertex = common.pop()
        center = vertices[vertex]
        directions = np.array([sign*(vertices[k]-center) for face,sign in ((i,-1),(j,1))
                               for k in faces[face] if k!=vertex])
        directions /= np.linalg.norm(directions,axis=1)[:,None]
        # A rotation-invariant maximum-margin cut treats repeated sectors alike.
        fit = minimize(lambda n: 0.5*(n@n), np.zeros(3), jac=lambda n:n,
                       constraints=[dict(type="ineq", fun=lambda n:directions@n-1,
                                         jac=lambda n:directions)], method="SLSQP")
        if not fit.success:
            raise ValueError(f"No separating corner cut for panels {i+1} and {j+1}")
        normal = fit.x / np.linalg.norm(fit.x)
        for k, sign in ((i,1),(j,-1)):
            rotation,_ = poses[k]
            local_normal = sign*(normal@rotation)
            corner_xy = panels[k][faces[k].index(vertex)]
            plane = np.r_[local_normal,local_normal[:2]@corner_xy]
            name,_,_,planes = groups[k]
            groups[k] = (name,*clipped_panel(np.vstack((planes,plane))))
            solids[k] = positioned(k)
        if penetrates(solids[i],solids[j]):
            raise ValueError("Corner relief did not clear the interference")
        count += 1
    return count


def relieve_flat_overlaps(groups, panels):
    """Trim protruding miter ends where separate branches meet in the flat net."""
    count = 0
    for i,j in itertools.combinations(range(len(groups)),2):
        if not penetrates(groups[i][1],groups[j][1]):
            continue
        a,b = panels[i],panels[j]
        edges = np.vstack((np.roll(a,-1,axis=0)-a,np.roll(b,-1,axis=0)-b))
        axes = edges[:,[1,0]]*[1,-1]
        axes /= np.linalg.norm(axes,axis=1)[:,None]
        axes = np.vstack((axes,-axes))
        high,low = (a@axes.T).max(axis=0),(b@axes.T).min(axis=0)
        best = np.argmax(low-high)
        if low[best] < high[best]-1e-6:
            raise ValueError("Compensated inner panels overlap; choose another net")
        plane = np.r_[axes[best],0,(low[best]+high[best])/2]
        for k,sign in ((i,1),(j,-1)):
            name,points,_,planes = groups[k]
            cut = sign*plane
            if np.max(points@cut[:3]-cut[3])>1e-6:
                groups[k] = (name,*clipped_panel(np.vstack((planes,cut))))
        count += 1
    return count


def fold_poses(vertices, faces, panels, incidence, angles, outer_axis_depth=0):
    """Anchor the floor, then rotate children about their parent's oriented edge."""
    layout = [np.column_stack((p, np.zeros(len(p)))) for p in panels]

    def frame(points):
        x = points[1]-points[0]
        x = x/np.linalg.norm(x)
        z = np.cross(x, points[2]-points[0])
        z /= np.linalg.norm(z)
        return np.column_stack((x, np.cross(z,x), z))

    floor = vertices[faces[0]]
    base = frame(floor) @ frame(layout[0]).T
    poses = {0: (base, floor[0]-base@layout[0][0])}
    adjacency = [[] for _ in faces]
    for key, angle in angles.items():
        (i,a,b),(j,c,d) = incidence[key]
        adjacency[i].append((j,a,b,angle))
        adjacency[j].append((i,c,d,angle))
    order, joints = [0], []
    for i in order:
        matrix, shift = poses[i]
        for j,a,b,angle in adjacency[i]:
            if j in poses:
                continue
            start,end = layout[i][[faces[i].index(a),faces[i].index(b)]]
            axis = end-start
            outward = np.array([axis[1], -axis[0], 0.0])/np.linalg.norm(axis)
            if outward@(layout[i].mean(axis=0)-start)>0:
                outward = -outward
            depth = outer_axis_depth if angle > 0 else 0
            offset = outward*(depth*math.tan(angle/2))
            # Separate the inner edges in the flat blank. The extended outer
            # edges meet halfway and are the actual hinge at z=depth.
            child_start = layout[j][faces[j].index(a)]
            if outer_axis_depth:
                layout[j] += start-child_start+2*offset
            start = start+offset+[0,0,depth]
            end = end+offset+[0,0,depth]
            turn = Rotation.from_rotvec(axis*(angle/np.linalg.norm(axis))).as_matrix()
            poses[j] = matrix@turn, shift+matrix@(start-turn@start)
            joints.append((i+1,j+1,*start,*end,math.degrees(angle)))
            order.append(j)
    if len(poses) != len(faces):
        raise ValueError("Disconnected fold tree")
    error = max(np.linalg.norm(p@poses[i][0].T+poses[i][1]-vertices[faces[i]],axis=1).max()
                for i,p in enumerate(layout))
    if error > 1e-4:
        raise ValueError(f"Folded panel vertices differ from OBJ by {error:g} mm")
    return poses, joints, error, [p[:,:2] for p in layout]


def write_step(path, groups, poses=None):
    import gmsh

    gmsh.initialize([], readConfigFiles=False)
    try:
        gmsh.model.add("Grooved blank")
        occ = gmsh.model.occ
        for index, (name, points, _, planes) in enumerate(groups):
            # Local topology per body keeps adjacent panels independently selectable.
            vertices = [occ.addPoint(*point) for point in points]
            edges, surfaces = {}, []
            for plane in planes:
                normal = plane[:3]
                indices = np.flatnonzero(abs(points @ normal - plane[3]) < 1e-7)
                if len(indices) < 3 or np.linalg.matrix_rank(points[indices]-points[indices[0]],tol=1e-7)<2:
                    continue
                offsets = points[indices] - points[indices].mean(axis=0)
                u = offsets[np.argmax(np.linalg.norm(offsets, axis=1))]
                v = np.cross(normal / np.linalg.norm(normal), u)
                order = indices[np.argsort(np.arctan2(offsets @ v, offsets @ u))].tolist()
                loop = []
                for a, b in cyclic(order):
                    if (a,b) not in edges:
                        tag = occ.addLine(vertices[a], vertices[b])
                        edges[a,b], edges[b,a] = tag, -tag
                    loop.append(edges[a,b])
                wire = occ.addWire(loop, checkClosed=True)
                surfaces.append(occ.addPlaneSurface([wire]))
            body = occ.addVolume([occ.addSurfaceLoop(surfaces)])
            if poses is not None:
                matrix, shift = poses[index]
                affine = np.eye(4)
                affine[:3,:3], affine[:3,3] = matrix, shift
                occ.affineTransform([(3,body)], affine.ravel().tolist())
            if not math.isclose(occ.getMass(3,body), ConvexHull(points).volume, rel_tol=1e-6):
                raise ValueError(f"Invalid STEP volume for {name}")
        occ.synchronize()
        if len(gmsh.model.getEntities(3)) != len(groups):
            raise ValueError("Unexpected number of STEP bodies")
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def write_variant(folder, vertices, faces, layout, smooth):
    thickness = 3.0
    panels,hinges,incidence = layout
    normals = []
    for face in faces:
        a,b,c = vertices[face[:3]]
        normal = np.cross(b-a,c-a)
        normals.append(normal/np.linalg.norm(normal))
    angles, slopes = {}, {}
    for key in (incidence if smooth else hinges):
        (i,a,b),(j,c,d) = incidence[key]
        if (a,b) != (d,c):
            raise ValueError("Inconsistent OBJ winding")
        axis = vertices[b]-vertices[a]
        angle = math.atan2(axis @ np.cross(normals[i],normals[j])/np.linalg.norm(axis),
                           normals[i] @ normals[j])
        angle = 0.0 if abs(angle)<1e-8 else angle
        slope = math.tan((-angle if smooth else max(0,-angle))/2)
        if key in hinges:
            angles[key] = angle
        slopes[key] = slope
    poses, joints, error, panels = fold_poses(vertices, faces, panels, incidence, angles,
                                              thickness if smooth else 0)
    groups = []
    for i,(face,xy) in enumerate(zip(faces,panels),1):
        cuts = [slopes.get(tuple(sorted(e)),0) for e in cyclic(face)]
        groups.append((f"panel_{i}", *panel_mesh(xy,cuts,thickness)))
    folder.mkdir()
    relieved = relieve_corners(groups, poses, vertices, faces, panels)
    print(f"{relieved} corner pairs relieved")
    if smooth:
        print(f"{relieve_flat_overlaps(groups, panels)} flat overlaps relieved")
    write_step(folder/"flat.stp", groups)
    write_step(folder/"folded.stp", groups, poses)
    print(f"{len(groups)} solids; {len(joints)} hinges; max vertex error {error:.3g} mm")
    with (folder/"flat.obj").open("w",encoding="ascii") as obj:
        obj.write("# Flat HeatPro board, mm. Outside +Z. Ideal slit faces kept separate.\n")
        base = 1
        for name,pts,triangles,_ in groups:
            obj.write(f"o {name}\n")
            for x,y,z in pts:
                obj.write(f"v {x:.8f} {y:.8f} {z:.8f}\n")
            for a,b,c in triangles+base:
                obj.write(f"f {a} {b} {c}\n")
            base += len(pts)
    print(f"{folder.name}: {len(panels)} panels, "
          f"{sum(abs(s) > 0 for s in slopes.values())} beveled edges")


def generate(run):
    vertices, faces = read_surface(run/"best.obj")
    svg = ET.parse(run/"best_net.svg")
    layout = read_layout(svg.getroot(), vertices, faces)
    # Finish both variants before replacing outputs from an earlier run.
    with TemporaryDirectory(dir=run) as temporary:
        stage = Path(temporary)
        for variant in ("rough", "smooth"):
            write_variant(stage/variant, vertices, faces, layout, smooth=(variant=="smooth"))
        for variant in ("rough", "smooth"):
            destination = run/variant
            destination.mkdir(exist_ok=True)
            for source in (stage/variant).iterdir():
                source.replace(destination/source.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,help="folder with best.obj and best_net.svg")
    generate(parser.parse_args().run)
