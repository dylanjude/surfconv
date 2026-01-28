#!/usr/bin/env python3
"""
UGRID to VTK/STL Converter

Reads UGRID file and converts to VTK or STL format.
"""

import numpy as np
import sys
from vtk_writer import VTKWriter
from stl_writer import write_stl, quads_to_triangles


def read_ugrid(filename):
    """
    Read UGRID file
    
    Returns:
        vertices: Nx3 array of vertex coordinates  
        tri_conn: Mx3 array of triangle connectivity
        quad_conn: Kx4 array of quad connectivity
    """
    with open(filename, "r") as f:
        nv, n3f, n4f, n4, n5, n6, n8 = [int(xx) for xx in f.readline().split()]
        
        # Read vertices
        x = np.zeros((nv, 3), 'd')
        for i in range(nv):
            x[i, :] = [float(xx) for xx in f.readline().split()]
        
        # Read triangular faces
        conn3 = np.zeros((n3f, 3), 'i')
        for i in range(n3f):
            conn3[i, :] = [int(xx) for xx in f.readline().split()]
        
        # Read quad faces
        conn4 = np.zeros((n4f, 4), 'i')
        for i in range(n4f):
            conn4[i, :] = [int(xx) for xx in f.readline().split()]
    
    return x, conn3, conn4


def ugrid_to_vtk(ugrid_file, vtk_file, format='xml'):
    """Convert UGRID to VTK format"""
    print(f"Reading UGRID file: {ugrid_file}")
    vertices, tri_conn, quad_conn = read_ugrid(ugrid_file)
    
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(tri_conn)}")
    print(f"  Quads: {len(quad_conn)}")
    
    # Check for 1-based indexing
    min_idx = min(tri_conn.min() if len(tri_conn) > 0 else 1,
                  quad_conn.min() if len(quad_conn) > 0 else 1)
    
    if min_idx == 1:
        print("  Converting from 1-based to 0-based indexing")
        if len(tri_conn) > 0:
            tri_conn = tri_conn - 1
        if len(quad_conn) > 0:
            quad_conn = quad_conn - 1
    
    # Prepare cells and cell types
    cells = []
    cell_types = []
    
    # Add triangles
    for tri in tri_conn:
        cells.append(tri)
        cell_types.append(5)  # VTK_TRIANGLE
    
    # Add quads
    for quad in quad_conn:
        cells.append(quad)
        cell_types.append(9)  # VTK_QUAD
    
    print(f"  Total cells: {len(cells)}")
    
    # Write VTK
    print(f"\nWriting VTK file: {vtk_file}")
    writer = VTKWriter()
    writer.set_points(vertices)
    writer.set_cells(cells, cell_types)
    
    if format == 'xml':
        writer.write_vtu_xml(vtk_file)
    else:
        writer.write_vtk_legacy(vtk_file, title="UGRID Surface Mesh")
    
    print("Conversion complete!")


def ugrid_to_stl(ugrid_file, stl_file, format='binary'):
    """Convert UGRID to STL format"""
    print(f"Reading UGRID file: {ugrid_file}")
    vertices, tri_conn, quad_conn = read_ugrid(ugrid_file)
    
    print(f"  Vertices: {len(vertices)}")
    print(f"  Triangles: {len(tri_conn)}")
    print(f"  Quads: {len(quad_conn)}")
    
    # Combine triangles and convert quads
    all_triangles = tri_conn.copy()
    if len(quad_conn) > 0:
        quad_triangles = quads_to_triangles(quad_conn)
        all_triangles = np.vstack([all_triangles, quad_triangles])
        print(f"  Converted quads to {len(quad_triangles)} triangles")
    
    # Check for 1-based indexing
    if len(all_triangles) > 0 and all_triangles.min() == 1:
        print("  Converting from 1-based to 0-based indexing")
        all_triangles = all_triangles - 1
    
    print(f"  Total triangles: {len(all_triangles)}")
    
    # Write STL
    print(f"\nWriting STL file: {stl_file}")
    write_stl(vertices, all_triangles, stl_file, format=format,
              solid_name='mesh', header=f"Converted from {ugrid_file}")
    
    print("Conversion complete!")


def main():
    if len(sys.argv) < 2:
        print("UGRID Surface Mesh Converter")
        print("=" * 50)
        print("\nUsage:")
        print("  ugrid2vtk <input.ugrid> [output] [options]")
        print("\nOptions:")
        print("  --vtk          Output VTK format (default)")
        print("  --vtu          Output VTK XML format (.vtu)")
        print("  --stl          Output STL format")
        print("  --stl-ascii    Output ASCII STL format")
        print("\nExamples:")
        print("  ugrid2vtk mesh.ugrid                    # Creates mesh.vtu")
        print("  ugrid2vtk mesh.ugrid mesh.vtk --vtk     # Creates legacy VTK")
        print("  ugrid2vtk mesh.ugrid mesh.stl --stl     # Creates binary STL")
        print("  ugrid2vtk mesh.ugrid --stl-ascii        # Creates ASCII STL")
        sys.exit(1)
    
    # Parse arguments
    input_file = sys.argv[1]
    output_file = None
    output_format = 'vtu'  # default
    
    # Check for format flags
    for arg in sys.argv[2:]:
        if arg.startswith('--'):
            if arg == '--vtk':
                output_format = 'vtk'
            elif arg == '--vtu':
                output_format = 'vtu'
            elif arg == '--stl':
                output_format = 'stl'
            elif arg == '--stl-ascii':
                output_format = 'stl-ascii'
        else:
            output_file = arg
    
    # Generate output filename if not provided
    if output_file is None:
        base = input_file.rsplit('.', 1)[0]
        if output_format == 'vtk':
            output_file = base + '.vtk'
        elif output_format == 'vtu':
            output_file = base + '.vtu'
        elif output_format in ['stl', 'stl-ascii']:
            output_file = base + '.stl'
    
    # Convert
    if output_format in ['vtk', 'vtu']:
        fmt = 'legacy' if output_format == 'vtk' else 'xml'
        ugrid_to_vtk(input_file, output_file, format=fmt)
    elif output_format in ['stl', 'stl-ascii']:
        fmt = 'ascii' if output_format == 'stl-ascii' else 'binary'
        ugrid_to_stl(input_file, output_file, format=fmt)


if __name__ == "__main__":
    main()
