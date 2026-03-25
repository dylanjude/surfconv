#!/usr/bin/env python3
"""
SurfConv - Universal Surface Mesh Converter

Converts between surface mesh formats:
  - UGRID (.ugrid)
  - VTK Legacy (.vtk)
  - VTK XML (.vtu)
  - STL Binary/ASCII (.stl)
  - Pointwise FACET (.facet)

Usage:
  surfconv <input> <output> [options]
  surfconv <input> --format <fmt> [options]
"""

import numpy as np
import sys
import os
import struct
import re
from typing import Tuple, Optional, Dict, Any


class SurfaceMesh:
    """
    Internal representation of a surface mesh.

    Attributes:
        vertices: Nx3 array of vertex coordinates
        triangles: Mx3 array of triangle connectivity (0-based)
        quads: Kx4 array of quad connectivity (0-based)
    """

    def __init__(self):
        self.vertices = np.zeros((0, 3), dtype=np.float64)
        self.triangles = np.zeros((0, 3), dtype=np.int32)
        self.quads = np.zeros((0, 4), dtype=np.int32)

    def info(self) -> str:
        """Return mesh statistics as a string"""
        return (f"Vertices: {len(self.vertices)}, "
                f"Triangles: {len(self.triangles)}, "
                f"Quads: {len(self.quads)}")

    def all_triangles(self) -> np.ndarray:
        """Return all faces as triangles (quads split into 2 triangles each)"""
        if len(self.quads) == 0:
            return self.triangles.copy()

        quad_tris = np.zeros((len(self.quads) * 2, 3), dtype=np.int32)
        for i, quad in enumerate(self.quads):
            quad_tris[2*i] = [quad[0], quad[1], quad[2]]
            quad_tris[2*i + 1] = [quad[0], quad[2], quad[3]]

        if len(self.triangles) == 0:
            return quad_tris
        return np.vstack([self.triangles, quad_tris])


# =============================================================================
# READERS
# =============================================================================

def read_ugrid(filename: str) -> SurfaceMesh:
    """Read UGRID surface mesh format"""
    mesh = SurfaceMesh()

    with open(filename, 'r') as f:
        header = f.readline().split()
        nv = int(header[0])
        n3f = int(header[1])
        n4f = int(header[2])

        # Read vertices
        mesh.vertices = np.zeros((nv, 3), dtype=np.float64)
        for i in range(nv):
            mesh.vertices[i] = [float(x) for x in f.readline().split()]

        # Read triangles
        mesh.triangles = np.zeros((n3f, 3), dtype=np.int32)
        for i in range(n3f):
            mesh.triangles[i] = [int(x) for x in f.readline().split()]

        # Read quads
        mesh.quads = np.zeros((n4f, 4), dtype=np.int32)
        for i in range(n4f):
            mesh.quads[i] = [int(x) for x in f.readline().split()]

    # Convert 1-based to 0-based indexing if needed
    min_idx = float('inf')
    if len(mesh.triangles) > 0:
        min_idx = min(min_idx, mesh.triangles.min())
    if len(mesh.quads) > 0:
        min_idx = min(min_idx, mesh.quads.min())

    if min_idx == 1:
        if len(mesh.triangles) > 0:
            mesh.triangles -= 1
        if len(mesh.quads) > 0:
            mesh.quads -= 1

    return mesh


def read_vtk_legacy(filename: str) -> SurfaceMesh:
    """Read VTK Legacy ASCII format (.vtk)"""
    mesh = SurfaceMesh()

    with open(filename, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    i = 0

    # Skip header
    while i < len(lines) and not lines[i].startswith('POINTS'):
        i += 1

    # Read points
    parts = lines[i].split()
    n_points = int(parts[1])
    i += 1

    mesh.vertices = np.zeros((n_points, 3), dtype=np.float64)
    pt_idx = 0
    values = []
    while pt_idx < n_points:
        values.extend(lines[i].split())
        while len(values) >= 3:
            mesh.vertices[pt_idx] = [float(values[0]), float(values[1]), float(values[2])]
            values = values[3:]
            pt_idx += 1
        i += 1

    # Find CELLS section
    while i < len(lines) and not lines[i].startswith('CELLS'):
        i += 1

    parts = lines[i].split()
    n_cells = int(parts[1])
    i += 1

    # Read cells temporarily
    temp_cells = []
    for _ in range(n_cells):
        cell_data = [int(x) for x in lines[i].split()]
        n_verts = cell_data[0]
        temp_cells.append(cell_data[1:n_verts+1])
        i += 1

    # Find CELL_TYPES section
    while i < len(lines) and not lines[i].startswith('CELL_TYPES'):
        i += 1
    i += 1

    # Read cell types and categorize
    triangles = []
    quads = []
    for cell in temp_cells:
        if len(cell) == 3:
            triangles.append(cell)
        elif len(cell) == 4:
            quads.append(cell)

    if triangles:
        mesh.triangles = np.array(triangles, dtype=np.int32)
    if quads:
        mesh.quads = np.array(quads, dtype=np.int32)

    return mesh


def read_vtu_xml(filename: str) -> SurfaceMesh:
    """Read VTK XML format (.vtu)"""
    mesh = SurfaceMesh()

    with open(filename, 'r') as f:
        content = f.read()

    # Extract points
    points_match = re.search(
        r'<Points>.*?<DataArray[^>]*>(.*?)</DataArray>.*?</Points>',
        content, re.DOTALL
    )
    if points_match:
        point_data = points_match.group(1).split()
        n_points = len(point_data) // 3
        mesh.vertices = np.array([float(x) for x in point_data], dtype=np.float64).reshape(n_points, 3)

    # Extract connectivity
    conn_match = re.search(
        r'<DataArray[^>]*Name="connectivity"[^>]*>(.*?)</DataArray>',
        content, re.DOTALL
    )

    # Extract offsets
    offset_match = re.search(
        r'<DataArray[^>]*Name="offsets"[^>]*>(.*?)</DataArray>',
        content, re.DOTALL
    )

    # Extract types
    types_match = re.search(
        r'<DataArray[^>]*Name="types"[^>]*>(.*?)</DataArray>',
        content, re.DOTALL
    )

    if conn_match and offset_match and types_match:
        connectivity = [int(x) for x in conn_match.group(1).split()]
        offsets = [int(x) for x in offset_match.group(1).split()]
        types = [int(x) for x in types_match.group(1).split()]

        triangles = []
        quads = []

        prev_offset = 0
        for offset, cell_type in zip(offsets, types):
            cell = connectivity[prev_offset:offset]
            if cell_type == 5 or len(cell) == 3:  # VTK_TRIANGLE
                triangles.append(cell)
            elif cell_type == 9 or len(cell) == 4:  # VTK_QUAD
                quads.append(cell)
            prev_offset = offset

        if triangles:
            mesh.triangles = np.array(triangles, dtype=np.int32)
        if quads:
            mesh.quads = np.array(quads, dtype=np.int32)

    return mesh


def read_stl_ascii(filename: str) -> SurfaceMesh:
    """Read ASCII STL format"""
    mesh = SurfaceMesh()

    with open(filename, 'r') as f:
        content = f.read()

    # Extract all vertices from facets
    vertex_pattern = re.compile(r'vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)')
    matches = vertex_pattern.findall(content)

    all_vertices = np.array([[float(x), float(y), float(z)] for x, y, z in matches], dtype=np.float64)
    n_triangles = len(all_vertices) // 3

    # STL stores vertices directly per triangle, so we need to deduplicate
    unique_vertices, inverse = np.unique(all_vertices, axis=0, return_inverse=True)

    mesh.vertices = unique_vertices
    mesh.triangles = inverse.reshape(n_triangles, 3).astype(np.int32)

    return mesh


def read_stl_binary(filename: str) -> SurfaceMesh:
    """Read binary STL format"""
    mesh = SurfaceMesh()

    with open(filename, 'rb') as f:
        # Skip 80-byte header
        f.read(80)

        # Read number of triangles
        n_triangles = struct.unpack('<I', f.read(4))[0]

        all_vertices = np.zeros((n_triangles * 3, 3), dtype=np.float64)

        for i in range(n_triangles):
            # Skip normal (3 floats = 12 bytes)
            f.read(12)

            # Read 3 vertices (9 floats = 36 bytes)
            for j in range(3):
                v = struct.unpack('<fff', f.read(12))
                all_vertices[i*3 + j] = v

            # Skip attribute byte count
            f.read(2)

    # Deduplicate vertices
    unique_vertices, inverse = np.unique(all_vertices, axis=0, return_inverse=True)

    mesh.vertices = unique_vertices
    mesh.triangles = inverse.reshape(n_triangles, 3).astype(np.int32)

    return mesh


def read_stl(filename: str) -> SurfaceMesh:
    """Read STL format (auto-detect binary or ASCII)"""
    with open(filename, 'rb') as f:
        header = f.read(80)

    # Check if it's ASCII (starts with "solid")
    try:
        header_str = header.decode('ascii', errors='ignore').strip()
        if header_str.startswith('solid'):
            # Could still be binary with "solid" in header, check further
            with open(filename, 'r') as f:
                first_lines = f.read(1000)
                if 'facet normal' in first_lines:
                    return read_stl_ascii(filename)
    except:
        pass

    return read_stl_binary(filename)


def read_facet(filename: str) -> SurfaceMesh:
    """Read Pointwise FACET format"""
    mesh = SurfaceMesh()

    with open(filename, 'r') as f:
        lines = f.readlines()

    i = 0

    # Skip header: "FACET FILE...", "1", "Grid", "0, ..."
    # Then vertex count follows
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('0,'):
            # Next line is vertex count
            i += 1
            nv = int(lines[i].strip())
            i += 1
            break
        i += 1

    # Read vertices
    mesh.vertices = np.zeros((nv, 3), dtype=np.float64)
    for v_idx in range(nv):
        mesh.vertices[v_idx] = [float(x) for x in lines[i].split()]
        i += 1

    # Read element sections
    triangles = []
    quads = []

    while i < len(lines):
        line = lines[i].strip()

        if line == '1':  # Section marker
            i += 1
            if i >= len(lines):
                break
            section_type = lines[i].strip()
            i += 1
            if i >= len(lines):
                break

            # Parse count line "N M" where N is count, M is vertices per element
            parts = lines[i].strip().split()
            n_elements = int(parts[0])
            verts_per_elem = int(parts[1])
            i += 1

            for _ in range(n_elements):
                if i >= len(lines):
                    break
                elem_data = lines[i].strip().split()
                connectivity = [int(elem_data[j]) for j in range(verts_per_elem)]

                if verts_per_elem == 3:
                    triangles.append(connectivity)
                elif verts_per_elem == 4:
                    quads.append(connectivity)
                i += 1
        else:
            i += 1

    if triangles:
        mesh.triangles = np.array(triangles, dtype=np.int32)
    if quads:
        mesh.quads = np.array(quads, dtype=np.int32)

    # FACET uses 1-based indexing
    min_idx = float('inf')
    if len(mesh.triangles) > 0:
        min_idx = min(min_idx, mesh.triangles.min())
    if len(mesh.quads) > 0:
        min_idx = min(min_idx, mesh.quads.min())

    if min_idx == 1:
        if len(mesh.triangles) > 0:
            mesh.triangles -= 1
        if len(mesh.quads) > 0:
            mesh.quads -= 1

    return mesh


# =============================================================================
# WRITERS
# =============================================================================

def write_ugrid(mesh: SurfaceMesh, filename: str):
    """Write UGRID surface mesh format (1-based indexing)"""
    nv = len(mesh.vertices)
    n3f = len(mesh.triangles)
    n4f = len(mesh.quads)

    with open(filename, 'w') as f:
        # Header: nv n3f n4f n4 n5 n6 n8 (volume elements are 0)
        f.write(f"{nv} {n3f} {n4f} 0 0 0 0\n")

        # Vertices
        for v in mesh.vertices:
            f.write(f"{v[0]:.16e} {v[1]:.16e} {v[2]:.16e}\n")

        # Triangles (1-based)
        for tri in mesh.triangles:
            f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1}\n")

        # Quads (1-based)
        for quad in mesh.quads:
            f.write(f"{quad[0]+1} {quad[1]+1} {quad[2]+1} {quad[3]+1}\n")

    print(f"Wrote {filename}")


def write_vtk_legacy(mesh: SurfaceMesh, filename: str):
    """Write VTK Legacy ASCII format (.vtk)"""
    with open(filename, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Surface Mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # Points
        f.write(f"POINTS {len(mesh.vertices)} double\n")
        for v in mesh.vertices:
            f.write(f"{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n")

        # Cells
        n_cells = len(mesh.triangles) + len(mesh.quads)
        cell_size = len(mesh.triangles) * 4 + len(mesh.quads) * 5
        f.write(f"\nCELLS {n_cells} {cell_size}\n")

        for tri in mesh.triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")
        for quad in mesh.quads:
            f.write(f"4 {quad[0]} {quad[1]} {quad[2]} {quad[3]}\n")

        # Cell types
        f.write(f"\nCELL_TYPES {n_cells}\n")
        for _ in mesh.triangles:
            f.write("5\n")  # VTK_TRIANGLE
        for _ in mesh.quads:
            f.write("9\n")  # VTK_QUAD

    print(f"Wrote {filename}")


def write_vtu_xml(mesh: SurfaceMesh, filename: str):
    """Write VTK XML format (.vtu)"""
    n_cells = len(mesh.triangles) + len(mesh.quads)

    with open(filename, 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
        f.write('  <UnstructuredGrid>\n')
        f.write(f'    <Piece NumberOfPoints="{len(mesh.vertices)}" NumberOfCells="{n_cells}">\n')

        # Points
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
        for v in mesh.vertices:
            f.write(f'          {v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n')
        f.write('        </DataArray>\n')
        f.write('      </Points>\n')

        # Cells
        f.write('      <Cells>\n')

        # Connectivity
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        f.write('          ')
        for tri in mesh.triangles:
            f.write(f'{tri[0]} {tri[1]} {tri[2]} ')
        for quad in mesh.quads:
            f.write(f'{quad[0]} {quad[1]} {quad[2]} {quad[3]} ')
        f.write('\n        </DataArray>\n')

        # Offsets
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        f.write('          ')
        offset = 0
        for _ in mesh.triangles:
            offset += 3
            f.write(f'{offset} ')
        for _ in mesh.quads:
            offset += 4
            f.write(f'{offset} ')
        f.write('\n        </DataArray>\n')

        # Types
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        f.write('          ')
        for _ in mesh.triangles:
            f.write('5 ')
        for _ in mesh.quads:
            f.write('9 ')
        f.write('\n        </DataArray>\n')

        f.write('      </Cells>\n')
        f.write('    </Piece>\n')
        f.write('  </UnstructuredGrid>\n')
        f.write('</VTKFile>\n')

    print(f"Wrote {filename}")


def write_stl_ascii(mesh: SurfaceMesh, filename: str, solid_name: str = 'mesh'):
    """Write ASCII STL format (triangles only)"""
    triangles = mesh.all_triangles()

    with open(filename, 'w') as f:
        f.write(f"solid {solid_name}\n")

        for tri in triangles:
            v0, v1, v2 = mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]

            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")

        f.write(f"endsolid {solid_name}\n")

    print(f"Wrote {filename}")


def write_stl_binary(mesh: SurfaceMesh, filename: str, header: str = 'Binary STL'):
    """Write binary STL format (triangles only)"""
    triangles = mesh.all_triangles()

    header_bytes = header.encode('ascii')[:80].ljust(80, b'\0')

    with open(filename, 'wb') as f:
        f.write(header_bytes)
        f.write(struct.pack('<I', len(triangles)))

        for tri in triangles:
            v0, v1, v2 = mesh.vertices[tri[0]], mesh.vertices[tri[1]], mesh.vertices[tri[2]]

            # Compute normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm

            f.write(struct.pack('<fff', *normal.astype(np.float32)))
            f.write(struct.pack('<fff', *v0.astype(np.float32)))
            f.write(struct.pack('<fff', *v1.astype(np.float32)))
            f.write(struct.pack('<fff', *v2.astype(np.float32)))
            f.write(struct.pack('<H', 0))

    print(f"Wrote {filename}")


def write_facet(mesh: SurfaceMesh, filename: str):
    """Write Pointwise FACET format (1-based indexing)"""
    nv = len(mesh.vertices)
    n3f = len(mesh.triangles)
    n4f = len(mesh.quads)

    with open(filename, 'w') as f:
        # Header
        f.write("FACET FILE V3.0    exported from Pointwise by SurfConv\n")
        f.write("1\n")
        f.write("Grid\n")
        f.write("0, 0.00 0.00 0.00 0.00\n")
        f.write(f"{nv}\n")

        # Vertices
        for v in mesh.vertices:
            f.write(f"{v[0]:16.8e} {v[1]:16.8e} {v[2]:16.8e}\n")

        # Quads section (1-based)
        if n4f > 0:
            f.write("1\n")
            f.write("Quadrilaterals\n")
            f.write(f"{n4f} 4\n")
            for e, quad in enumerate(mesh.quads):
                f.write(f"{quad[0]+1} {quad[1]+1} {quad[2]+1} {quad[3]+1} 0 0001 {e}\n")

        # Triangles section (1-based)
        if n3f > 0:
            f.write("1\n")
            f.write("Triangles\n")
            f.write(f"{n3f} 3\n")
            for e, tri in enumerate(mesh.triangles):
                f.write(f"{tri[0]+1} {tri[1]+1} {tri[2]+1} 0 0001 {n4f+e}\n")

    print(f"Wrote {filename}")


def read_tecplot(filename: str) -> SurfaceMesh:
    """Read Tecplot ASCII finite element format (.dat) with multiple zones"""
    mesh = SurfaceMesh()

    with open(filename, 'r') as f:
        lines = f.readlines()

    all_vertices = []
    all_triangles = []
    all_quads = []
    i = 0

    # Process all zones
    while i < len(lines):
        line = lines[i].strip().upper()

        if line.startswith('ZONE'):
            # Parse zone parameters
            zone_line = lines[i]
            zone_upper = zone_line.upper()

            n_points = 0
            n_elements = 0
            elem_type = 'TRIANGLE'

            # Extract N (number of points)
            n_match = re.search(r'\bN\s*=\s*(\d+)', zone_upper)
            if n_match:
                n_points = int(n_match.group(1))

            # Extract E (number of elements)
            e_match = re.search(r'\bE\s*=\s*(\d+)', zone_upper)
            if e_match:
                n_elements = int(e_match.group(1))

            # Extract ET (element type)
            et_match = re.search(r'\bET\s*=\s*(\w+)', zone_upper)
            if et_match:
                elem_type = et_match.group(1)

            i += 1

            # Read vertex coordinates for this zone
            zone_vertices = []
            values = []
            while len(zone_vertices) < n_points and i < len(lines):
                line = lines[i].strip()
                if line and not line.upper().startswith('ZONE') and not line.startswith('#'):
                    values.extend(line.split())
                    while len(values) >= 3:
                        zone_vertices.append([float(values[0]), float(values[1]), float(values[2])])
                        values = values[3:]
                elif line.upper().startswith('ZONE'):
                    break
                i += 1

            # Calculate vertex offset for this zone's connectivity
            vertex_offset = len(all_vertices)

            # Check if vertices match existing (shared vertices between zones)
            vertices_match = False
            if len(all_vertices) == len(zone_vertices) and len(all_vertices) > 0:
                # Check if vertices are approximately the same
                existing = np.array(all_vertices)
                new = np.array(zone_vertices)
                if np.allclose(existing, new, rtol=1e-10):
                    vertices_match = True
                    vertex_offset = 0

            if not vertices_match:
                all_vertices.extend(zone_vertices)

            # Read element connectivity for this zone
            elem_idx = 0
            while elem_idx < n_elements and i < len(lines):
                line = lines[i].strip()
                if line and not line.upper().startswith('ZONE') and not line.startswith('#'):
                    parts = [int(x) for x in line.split()]
                    # Adjust for vertex offset (connectivity is 1-based, we'll convert later)
                    if elem_type.startswith('QUAD') or len(parts) >= 4:
                        all_quads.append([p + vertex_offset for p in parts[:4]])
                    else:
                        all_triangles.append([p + vertex_offset for p in parts[:3]])
                    elem_idx += 1
                elif line.upper().startswith('ZONE'):
                    break
                i += 1
        else:
            i += 1

    # Build final mesh
    if all_vertices:
        mesh.vertices = np.array(all_vertices, dtype=np.float64)
    if all_triangles:
        mesh.triangles = np.array(all_triangles, dtype=np.int32)
    if all_quads:
        mesh.quads = np.array(all_quads, dtype=np.int32)

    # Convert 1-based to 0-based indexing if needed
    min_idx = float('inf')
    if len(mesh.triangles) > 0:
        min_idx = min(min_idx, mesh.triangles.min())
    if len(mesh.quads) > 0:
        min_idx = min(min_idx, mesh.quads.min())

    if min_idx == 1:
        if len(mesh.triangles) > 0:
            mesh.triangles -= 1
        if len(mesh.quads) > 0:
            mesh.quads -= 1

    return mesh


def write_tecplot(mesh: SurfaceMesh, filename: str):
    """Write Tecplot ASCII finite element format (.dat)"""
    nv = len(mesh.vertices)
    n3f = len(mesh.triangles)
    n4f = len(mesh.quads)

    with open(filename, 'w') as f:
        f.write('TITLE = "Surface Mesh"\n')
        f.write('VARIABLES = "X", "Y", "Z"\n')

        # Write triangles zone if present
        if n3f > 0:
            f.write(f'ZONE T="Triangles", N={nv}, E={n3f}, F=FEPOINT, ET=TRIANGLE\n')

            # Write vertices
            for v in mesh.vertices:
                f.write(f'{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n')

            # Write connectivity (1-based)
            for tri in mesh.triangles:
                f.write(f'{tri[0]+1} {tri[1]+1} {tri[2]+1}\n')

        # Write quads zone if present
        if n4f > 0:
            if n3f > 0:
                # Second zone shares vertices, but Tecplot needs them repeated
                f.write(f'ZONE T="Quads", N={nv}, E={n4f}, F=FEPOINT, ET=QUADRILATERAL\n')
                for v in mesh.vertices:
                    f.write(f'{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n')
            else:
                f.write(f'ZONE T="Quads", N={nv}, E={n4f}, F=FEPOINT, ET=QUADRILATERAL\n')
                for v in mesh.vertices:
                    f.write(f'{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n')

            # Write connectivity (1-based)
            for quad in mesh.quads:
                f.write(f'{quad[0]+1} {quad[1]+1} {quad[2]+1} {quad[3]+1}\n')

        # If no elements, just write vertices as a point zone
        if n3f == 0 and n4f == 0:
            f.write(f'ZONE T="Points", I={nv}, F=POINT\n')
            for v in mesh.vertices:
                f.write(f'{v[0]:.10e} {v[1]:.10e} {v[2]:.10e}\n')

    print(f"Wrote {filename}")


def _is_farfield_patch(patch_type):
    """Return True if a boundary patch looks like a non-physical boundary."""
    ptype = patch_type.lower().replace('-', '').replace(' ', '')
    for tag in ('farfield', 'freestream', 'overset'):
        if tag in ptype:
            return True
    return False


def _read_avm_str(buf):
    """Decode a fixed-length AVM string field, stripping null padding."""
    return buf.split(b'\x00', 1)[0].decode('ascii', errors='replace')


def read_avm(filename: str) -> SurfaceMesh:
    """Read AVM surface mesh (physical boundary faces only from unstruc meshes).

    Pure-Python binary reader — no compiled avmeshlib dependency needed.
    Far-field / freestream patches are automatically excluded so the result
    contains only body-surface faces.
    """
    mesh = SurfaceMesh()

    with open(filename, 'rb') as f:
        # --- File ID prefix (14 bytes) ---
        magic_str = f.read(6)
        if magic_str != b'AVMESH':
            raise ValueError(f"Not an AVM file (bad magic): {filename}")

        magic_num = struct.unpack('<i', f.read(4))[0]
        if magic_num == 1:
            endian = '<'
        else:
            # Try big-endian
            magic_num = struct.unpack('>i', struct.pack('<i', magic_num))[0]
            if magic_num == 1:
                endian = '>'
            else:
                raise ValueError(f"Cannot determine endianness of AVM file")

        I = endian + 'i'   # int format
        D = endian + 'd'   # double format

        revision = struct.unpack(I, f.read(4))[0]

        # --- File header ---
        # Rev0 has a different layout; rev1 and rev2 share the same file header
        if revision == 0:
            f.read(128)  # meshRevision
            meshCount = struct.unpack(I, f.read(4))[0]
            f.read(128)  # contactName
            f.read(128)  # contactOrg
            f.read(128)  # contactEmail
            precision = struct.unpack(I, f.read(4))[0]
            dimensions = struct.unpack(I, f.read(4))[0]
            desc_size = struct.unpack(I, f.read(4))[0]
            f.read(desc_size)  # description
        else:
            meshCount = struct.unpack(I, f.read(4))[0]
            f.read(128)  # contactInfo
            precision = struct.unpack(I, f.read(4))[0]
            f.read(4)    # dimensions
            desc_size = struct.unpack(I, f.read(4))[0]
            f.read(desc_size)  # description

        coord_bytes = 4 if precision == 1 else 8

        all_vertices = []
        all_triangles = []
        all_quads = []
        vertex_offset = 0

        # --- Read all mesh headers, then data sequentially ---
        # We need to store header info for all meshes first, because AVM
        # stores all headers before data.
        mesh_infos = []

        for meshid in range(meshCount):
            # Generic mesh header (layout differs by revision)
            if revision == 0:
                meshName = _read_avm_str(f.read(128))
                meshType = _read_avm_str(f.read(128))
                f.read(128)  # meshGenerator
                f.read(128)  # changedDate
                f.read(128)  # coordinateSystem
                f.read(8)    # modelScale
                f.read(128)  # gridUnits
                f.read(8)    # reynoldsNumber
                f.read(8)    # referenceLength
                f.read(8)    # wallDistance
                f.read(24)   # referencePoint[3]
                f.read(128)  # referencePointDescription
                f.read(8)    # periodicity
                f.read(128)  # periodicityDescription
                f.read(4)    # refinementLevel
                mdesc_size = struct.unpack(I, f.read(4))[0]
                f.read(mdesc_size)  # description
            elif revision == 1:
                meshName = _read_avm_str(f.read(128))
                meshType = _read_avm_str(f.read(128))
                f.read(128)  # meshGenerator
                f.read(128)  # coordinateSystem
                f.read(8)    # modelScale
                f.read(128)  # gridUnits
                f.read(8)    # referenceLength
                f.read(8)    # referenceArea
                f.read(24)   # referencePoint[3]
                f.read(128)  # referencePointDescription
                f.read(128)  # meshDescription
            else:  # rev2
                meshName = _read_avm_str(f.read(128))
                meshType = _read_avm_str(f.read(128))
                f.read(128)  # meshGenerator
                f.read(128)  # coordinateSystem
                f.read(8)    # modelScale
                f.read(128)  # gridUnits
                f.read(24)   # referenceLength[3]
                f.read(8)    # referenceArea
                f.read(24)   # referencePoint[3]
                f.read(128)  # referencePointDescription
                f.read(4)    # refined
                f.read(128)  # meshDescription

            info = {'name': meshName, 'type': meshType}

            # Type-specific header
            if meshType == 'unstruc':
                if revision == 0:
                    hdr_fields = struct.unpack(endian + '20i', f.read(80))
                    info['nNodes']         = hdr_fields[2]
                    info['nTriFaces']      = hdr_fields[6]
                    info['nQuadFaces']     = hdr_fields[7]
                    info['nBndTriFaces']   = hdr_fields[8]
                    info['nBndQuadFaces']  = hdr_fields[9]
                    info['nCells']         = hdr_fields[10]
                    info['nHexCells']      = hdr_fields[13]
                    info['nTetCells']      = hdr_fields[14]
                    info['nPriCells']      = hdr_fields[15]
                    info['nPyrCells']      = hdr_fields[16]
                    info['nPatches']       = hdr_fields[17]
                    info['nEdges']         = 0
                    info['nNodesOnGeometry'] = 0
                    info['nEdgesOnGeometry'] = 0
                    info['nFacesOnGeometry'] = 0
                    info['nPolyCells']     = 0
                    info['nPolyFaces']     = 0
                    info['nBndPolyFaces']  = 0
                    info['bndPolyFacesSize'] = 0
                    info['polyFacesSize']  = 0
                elif revision == 1:
                    hdr_fields = struct.unpack(endian + '25i', f.read(100))
                    info['nNodes']         = hdr_fields[0]
                    info['nCells']         = hdr_fields[2]
                    info['nPatches']       = hdr_fields[6]
                    info['nHexCells']      = hdr_fields[7]
                    info['nTetCells']      = hdr_fields[8]
                    info['nPriCells']      = hdr_fields[9]
                    info['nPyrCells']      = hdr_fields[10]
                    info['nPolyCells']     = hdr_fields[11]
                    info['nBndTriFaces']   = hdr_fields[12]
                    info['nTriFaces']      = hdr_fields[13]
                    info['nBndQuadFaces']  = hdr_fields[14]
                    info['nQuadFaces']     = hdr_fields[15]
                    info['nBndPolyFaces']  = hdr_fields[16]
                    info['nPolyFaces']     = hdr_fields[17]
                    info['bndPolyFacesSize'] = hdr_fields[18]
                    info['polyFacesSize']  = hdr_fields[19]
                    info['nEdges']         = hdr_fields[20]
                    info['nNodesOnGeometry'] = hdr_fields[21]
                    info['nEdgesOnGeometry'] = hdr_fields[22]
                    info['nFacesOnGeometry'] = hdr_fields[23]
                else:  # rev2
                    # 6 ints, 32-byte string, then 16 more ints
                    first6 = struct.unpack(endian + '6i', f.read(24))
                    f.read(32)  # elementScheme
                    rest = struct.unpack(endian + '16i', f.read(64))
                    info['nNodes']         = first6[0]
                    info['nCells']         = first6[2]
                    info['facePolyOrder']  = rest[0]
                    info['cellPolyOrder']  = rest[1]
                    info['nPatches']       = rest[2]
                    info['nHexCells']      = rest[3]
                    info['nTetCells']      = rest[4]
                    info['nPriCells']      = rest[5]
                    info['nPyrCells']      = rest[6]
                    info['nBndTriFaces']   = rest[7]
                    info['nTriFaces']      = rest[8]
                    info['nBndQuadFaces']  = rest[9]
                    info['nQuadFaces']     = rest[10]
                    info['nEdges']         = rest[11]
                    info['nNodesOnGeometry'] = rest[12]
                    info['nEdgesOnGeometry'] = rest[13]
                    info['nFacesOnGeometry'] = rest[14]
                    # rest[15] = geomRegionId (unused)
                    info['nPolyCells']     = 0
                    info['nPolyFaces']     = 0
                    info['nBndPolyFaces']  = 0
                    info['bndPolyFacesSize'] = 0
                    info['polyFacesSize']  = 0

                # Read patch headers (48 bytes each: 32s label + 16s type + int id)
                patches = []
                skip_patches = set()
                for _ in range(info['nPatches']):
                    plabel = _read_avm_str(f.read(32))
                    ptype = _read_avm_str(f.read(16))
                    patch_id = struct.unpack(I, f.read(4))[0]
                    patches.append((plabel, ptype, patch_id))
                    if _is_farfield_patch(ptype):
                        skip_patches.add(abs(patch_id))
                        print(f"  Skipping far-field patch: {plabel!r} "
                              f"(type={ptype!r})")
                info['patches'] = patches
                info['skip_patches'] = skip_patches

            elif meshType == 'bfstruc':
                if revision == 0:
                    f.read(8)    # jmax, kmax
                    f.read(4)    # lmax ... (need to skip the right amount)
                    # This is complex; for now just record type to skip later
                    pass
                # Skip bfstruc headers — we only care about unstruc
                # The exact size depends on revision; handled in data skip below
                pass

            # For non-unstruc types, we'd need to skip their type-specific
            # headers. For simplicity, assume all meshes are unstruc.
            # TODO: handle bfstruc/strand/cart header skipping if needed

            mesh_infos.append(info)

        # --- Read mesh data ---
        for info in mesh_infos:
            if info['type'] != 'unstruc':
                # Skip non-unstruc mesh data (not implemented)
                continue

            h = info
            nNodes = h['nNodes']
            skip = h['skip_patches']

            # Read nodes
            node_data = f.read(3 * nNodes * coord_bytes)
            fmt = endian + str(3 * nNodes) + ('f' if precision == 1 else 'd')
            coords = struct.unpack(fmt, node_data)
            verts = np.array(coords, dtype=np.float64).reshape(nNodes, 3)
            all_vertices.append(verts)

            # Face strides depend on facePolyOrder:
            #   nodesPerTri = (p+1)*(p+2)/2, nodesPerQuad = (p+1)^2
            #   stride = nodesPerFace + 1 (for cell ID)
            # For linear (p=1): tri=4 (3 nodes + cellID), quad=5 (4 nodes + cellID)
            fpo = h.get('facePolyOrder', 1)
            nodes_per_tri = ((fpo + 1) * (fpo + 2)) // 2
            nodes_per_quad = (fpo + 1) * (fpo + 1)
            tri_stride = nodes_per_tri + 1
            quad_stride = nodes_per_quad + 1

            # Read boundary tri faces
            # Format: n1, n2, ..., nN, cellID  (boundary faces written first)
            # Boundary faces have negative cellID = -(patch_id)
            nBndTri = h['nBndTriFaces']
            if nBndTri > 0:
                buf = f.read(tri_stride * nBndTri * 4)
                raw = struct.unpack(endian + str(tri_stride * nBndTri) + 'i', buf)
                kept = []
                for i in range(nBndTri):
                    base = i * tri_stride
                    n1, n2, n3 = raw[base], raw[base+1], raw[base+2]
                    cell_id = raw[base + tri_stride - 1]  # last value
                    patch_id = abs(cell_id)
                    if patch_id not in skip:
                        kept.append([n1 - 1 + vertex_offset,
                                     n2 - 1 + vertex_offset,
                                     n3 - 1 + vertex_offset])
                if kept:
                    all_triangles.append(np.array(kept, dtype=np.int32))

            # Read boundary quad faces
            nBndQuad = h['nBndQuadFaces']
            if nBndQuad > 0:
                buf = f.read(quad_stride * nBndQuad * 4)
                raw = struct.unpack(endian + str(quad_stride * nBndQuad) + 'i', buf)
                kept = []
                for i in range(nBndQuad):
                    base = i * quad_stride
                    n1, n2, n3, n4 = raw[base], raw[base+1], raw[base+2], raw[base+3]
                    cell_id = raw[base + quad_stride - 1]  # last value
                    patch_id = abs(cell_id)
                    if patch_id not in skip:
                        kept.append([n1 - 1 + vertex_offset,
                                     n2 - 1 + vertex_offset,
                                     n3 - 1 + vertex_offset,
                                     n4 - 1 + vertex_offset])
                if kept:
                    all_quads.append(np.array(kept, dtype=np.int32))

            # Skip remaining data: interior faces, cells, edges, geometry
            nIntTri = h['nTriFaces'] - nBndTri
            nIntQuad = h['nQuadFaces'] - nBndQuad

            # Cell strides depend on cellPolyOrder
            cpo = h.get('cellPolyOrder', 1)
            nph = (cpo + 1) ** 3                               # hex
            npt = ((cpo + 1) * (cpo + 2) * (cpo + 3)) // 6    # tet
            npp = ((cpo + 1) * (cpo + 1) * (cpo + 2)) // 2    # pri
            npy = ((cpo + 1) * (cpo + 2) * (2*cpo + 3)) // 6  # pyr

            skip_bytes = (
                tri_stride * nIntTri * 4 +                   # interior tris
                quad_stride * nIntQuad * 4 +                 # interior quads
                h.get('bndPolyFacesSize', 0) * 4 +           # bnd polys
                (h.get('polyFacesSize', 0) -
                 h.get('bndPolyFacesSize', 0)) * 4 +         # int polys
                nph * h['nHexCells'] * 4 +                   # hex cells
                npt * h['nTetCells'] * 4 +                   # tet cells
                npp * h['nPriCells'] * 4 +                   # pri cells
                npy * h['nPyrCells'] * 4 +                   # pyr cells
                2 * h['nEdges'] * 4                          # edges
            )
            # AMR/geometry data (node: 2 ints + 1 double = 16 bytes)
            skip_bytes += h['nNodesOnGeometry'] * 16
            skip_bytes += 3 * h['nEdgesOnGeometry'] * 4
            skip_bytes += 3 * h['nFacesOnGeometry'] * 4
            # Volume IDs
            skip_bytes += (h['nHexCells'] + h['nTetCells'] +
                           h['nPriCells'] + h['nPyrCells']) * 4
            f.seek(skip_bytes, 1)

            vertex_offset += nNodes

    # Combine all meshes
    if all_vertices:
        all_verts = np.vstack(all_vertices)
    else:
        all_verts = np.zeros((0, 3), dtype=np.float64)
    if all_triangles:
        all_tris = np.vstack(all_triangles)
    else:
        all_tris = np.zeros((0, 3), dtype=np.int32)
    if all_quads:
        all_quads_arr = np.vstack(all_quads)
    else:
        all_quads_arr = np.zeros((0, 4), dtype=np.int32)

    # Compact vertices — only keep nodes referenced by surface faces
    used = np.unique(np.concatenate([all_tris.ravel(), all_quads_arr.ravel()]))
    old_to_new = np.full(len(all_verts), -1, dtype=np.int32)
    old_to_new[used] = np.arange(len(used), dtype=np.int32)
    mesh.vertices = all_verts[used]
    if len(all_tris) > 0:
        mesh.triangles = old_to_new[all_tris]
    if len(all_quads_arr) > 0:
        mesh.quads = old_to_new[all_quads_arr]

    return mesh


# =============================================================================
# FORMAT DETECTION AND DISPATCH
# =============================================================================

FORMATS = {
    'ugrid': {'ext': '.ugrid', 'read': read_ugrid, 'write': write_ugrid},
    'vtk': {'ext': '.vtk', 'read': read_vtk_legacy, 'write': write_vtk_legacy},
    'vtu': {'ext': '.vtu', 'read': read_vtu_xml, 'write': write_vtu_xml},
    'stl': {'ext': '.stl', 'read': read_stl, 'write': write_stl_ascii},
    'stl-binary': {'ext': '.stl', 'read': read_stl, 'write': write_stl_binary},
    'facet': {'ext': '.facet', 'read': read_facet, 'write': write_facet},
    'tecplot': {'ext': '.dat', 'read': read_tecplot, 'write': write_tecplot},
    'avm': {'ext': '.avm', 'read': read_avm, 'write': None},
}


def detect_format(filename: str) -> Optional[str]:
    """Detect format from file extension"""
    ext = os.path.splitext(filename)[1].lower()
    for fmt, info in FORMATS.items():
        if info['ext'] == ext:
            return fmt
    return None


def read_mesh(filename: str, format: Optional[str] = None) -> SurfaceMesh:
    """Read mesh from file, auto-detecting format if not specified"""
    if format is None:
        format = detect_format(filename)

    if format is None:
        raise ValueError(f"Cannot detect format for {filename}")

    # Normalize format name
    if format == 'stl-ascii':
        format = 'stl'

    if format not in FORMATS:
        raise ValueError(f"Unknown format: {format}")

    print(f"Reading {format.upper()} file: {filename}")
    mesh = FORMATS[format]['read'](filename)
    print(f"  {mesh.info()}")
    return mesh


def write_mesh(mesh: SurfaceMesh, filename: str, format: Optional[str] = None):
    """Write mesh to file, auto-detecting format if not specified"""
    if format is None:
        format = detect_format(filename)

    if format is None:
        raise ValueError(f"Cannot detect format for {filename}")

    if format not in FORMATS:
        raise ValueError(f"Unknown format: {format}")

    if FORMATS[format]['write'] is None:
        raise ValueError(f"Writing {format.upper()} format is not supported (read-only)")

    if format == 'stl' and len(mesh.quads) > 0:
        n_quad_tris = len(mesh.quads) * 2
        print(f"  Note: Converting {len(mesh.quads)} quads to {n_quad_tris} triangles for STL")

    print(f"Writing {format.upper()} file: {filename}")
    FORMATS[format]['write'](mesh, filename)


def convert(input_file: str, output_file: str,
            input_format: Optional[str] = None,
            output_format: Optional[str] = None):
    """Convert mesh from input to output format"""
    mesh = read_mesh(input_file, input_format)
    write_mesh(mesh, output_file, output_format)
    print("Conversion complete!")


# =============================================================================
# CLI
# =============================================================================

def print_usage():
    print("SurfConv - Universal Surface Mesh Converter")
    print("=" * 50)
    print("\nUsage:")
    print("  surfconv <input> <output>")
    print("  surfconv <input> --format <fmt>")
    print("\nSupported formats:")
    print("  ugrid      UGRID surface mesh (.ugrid)")
    print("  vtk        VTK Legacy ASCII (.vtk)")
    print("  vtu        VTK XML (.vtu)")
    print("  stl        STL ASCII (.stl)")
    print("  stl-binary STL binary (.stl)")
    print("  facet      Pointwise FACET (.facet)")
    print("  tecplot    Tecplot ASCII FE (.dat)")
    print("  avm        AVMesh surface (.avm) [read-only]")
    print("\nOptions:")
    print("  --format <fmt>   Output format (auto-generates filename)")
    print("  --input-format   Override input format detection")
    print("  --output-format  Override output format detection")
    print("\nExamples:")
    print("  surfconv mesh.ugrid mesh.vtu          # UGRID to VTU")
    print("  surfconv mesh.ugrid --format stl      # UGRID to STL (mesh.stl)")
    print("  surfconv mesh.stl mesh.ugrid          # STL to UGRID")
    print("  surfconv mesh.vtu mesh.facet          # VTU to FACET")
    print("  surfconv a.stl b.vtk --output-format vtk")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    # Parse arguments
    args = sys.argv[1:]
    input_file = None
    output_file = None
    output_format = None
    input_format = None

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == '--format':
            i += 1
            output_format = args[i]
        elif arg == '--input-format':
            i += 1
            input_format = args[i]
        elif arg == '--output-format':
            i += 1
            output_format = args[i]
        elif arg in ['-h', '--help']:
            print_usage()
            sys.exit(0)
        elif input_file is None:
            input_file = arg
        else:
            output_file = arg
        i += 1

    if input_file is None:
        print("Error: No input file specified")
        sys.exit(1)

    # Generate output filename if not provided
    if output_file is None:
        if output_format is None:
            print("Error: No output file or format specified")
            sys.exit(1)

        base = os.path.splitext(input_file)[0]
        ext = FORMATS[output_format]['ext']
        output_file = base + ext

    # Auto-detect output format from filename if not specified
    if output_format is None:
        output_format = detect_format(output_file)

    convert(input_file, output_file, input_format, output_format)


if __name__ == "__main__":
    main()
