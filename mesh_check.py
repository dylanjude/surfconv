#!/usr/bin/env python3
"""
Mesh Topology Checker

Checks surface mesh triangles for consistent orientation and manifold properties.

Checks performed:
  1. Edge consistency - each interior edge shared by exactly 2 triangles
  2. Orientation consistency - adjacent triangles have compatible normals
  3. Non-manifold detection - edges shared by more than 2 triangles
  4. Boundary edge detection - edges with only 1 triangle

Usage:
  mesh_check.py <input_file>
"""

import sys
import numpy as np
from collections import defaultdict
from surfconv import read_mesh


def make_edge(v1, v2):
    """Create a canonical (sorted) edge tuple for edge counting"""
    return (min(v1, v2), max(v1, v2))


def make_directed_edge(v1, v2):
    """Create a directed edge tuple (preserves order)"""
    return (v1, v2)


def check_mesh_topology(mesh):
    """
    Check mesh topology for orientation consistency.

    Returns a dict with:
        - consistent: bool, True if all orientations are consistent
        - n_boundary_edges: number of boundary edges (1 triangle)
        - n_interior_edges: number of interior edges (2 triangles)
        - n_nonmanifold_edges: number of non-manifold edges (>2 triangles)
        - n_inconsistent_edges: number of edges with inconsistent orientation
        - boundary_edges: list of boundary edge tuples
        - nonmanifold_edges: list of non-manifold edge tuples
        - inconsistent_edges: list of inconsistently oriented edge tuples
    """
    triangles = mesh.all_triangles()

    # Track directed edges: maps directed_edge -> list of triangle indices
    directed_edges = defaultdict(list)

    # Track undirected edges: maps edge -> list of (tri_idx, directed_edge)
    edge_triangles = defaultdict(list)

    for tri_idx, tri in enumerate(triangles):
        # Each triangle has 3 edges
        edges = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0])
        ]

        for v1, v2 in edges:
            directed_edges[(v1, v2)].append(tri_idx)
            canonical = make_edge(v1, v2)
            edge_triangles[canonical].append((tri_idx, (v1, v2)))

    # Analyze edges
    boundary_edges = []
    interior_edges = []
    nonmanifold_edges = []
    inconsistent_edges = []

    for edge, tri_list in edge_triangles.items():
        n_tris = len(tri_list)

        if n_tris == 1:
            boundary_edges.append(edge)
        elif n_tris == 2:
            interior_edges.append(edge)

            # Check orientation consistency
            # For consistent orientation, the two triangles should traverse
            # the shared edge in opposite directions
            dir1 = tri_list[0][1]
            dir2 = tri_list[1][1]

            # If both have same direction, orientation is inconsistent
            if dir1 == dir2:
                inconsistent_edges.append(edge)
            # Check if they're properly opposite (v1,v2) vs (v2,v1)
            elif dir1 != (dir2[1], dir2[0]):
                inconsistent_edges.append(edge)
        else:
            nonmanifold_edges.append(edge)

    is_consistent = len(inconsistent_edges) == 0 and len(nonmanifold_edges) == 0

    return {
        'consistent': is_consistent,
        'n_triangles': len(triangles),
        'n_edges': len(edge_triangles),
        'n_boundary_edges': len(boundary_edges),
        'n_interior_edges': len(interior_edges),
        'n_nonmanifold_edges': len(nonmanifold_edges),
        'n_inconsistent_edges': len(inconsistent_edges),
        'boundary_edges': boundary_edges,
        'nonmanifold_edges': nonmanifold_edges,
        'inconsistent_edges': inconsistent_edges,
    }


def check_euler_characteristic(mesh, result):
    """
    Compute Euler characteristic: V - E + F

    For closed surfaces:
        Sphere: chi = 2
        Torus: chi = 0

    For surfaces with boundary, chi depends on topology.
    """
    V = len(mesh.vertices)
    E = result['n_edges']
    F = result['n_triangles']

    chi = V - E + F
    return chi


def print_report(mesh, result):
    """Print a human-readable topology report"""
    print("=" * 60)
    print("MESH TOPOLOGY CHECK")
    print("=" * 60)

    print(f"\nMesh Statistics:")
    print(f"  Vertices:  {len(mesh.vertices)}")
    print(f"  Triangles: {result['n_triangles']}")
    print(f"  Edges:     {result['n_edges']}")

    if len(mesh.quads) > 0:
        print(f"  (Note: {len(mesh.quads)} quads were split into {len(mesh.quads)*2} triangles)")

    print(f"\nEdge Analysis:")
    print(f"  Interior edges (2 triangles): {result['n_interior_edges']}")
    print(f"  Boundary edges (1 triangle):  {result['n_boundary_edges']}")
    print(f"  Non-manifold edges (>2 tri):  {result['n_nonmanifold_edges']}")

    chi = check_euler_characteristic(mesh, result)
    print(f"\nEuler Characteristic (V - E + F): {chi}")
    if result['n_boundary_edges'] == 0:
        if chi == 2:
            print("  -> Consistent with closed sphere-like surface")
        elif chi == 0:
            print("  -> Consistent with closed torus-like surface")
        else:
            print(f"  -> Genus approximately {(2 - chi) // 2}")
    else:
        print(f"  -> Surface has {result['n_boundary_edges']} boundary edges (open surface)")

    print(f"\nOrientation Check:")
    if result['n_inconsistent_edges'] == 0:
        print("  All triangle orientations are CONSISTENT")
    else:
        print(f"  WARNING: {result['n_inconsistent_edges']} edges have INCONSISTENT orientation")
        if result['n_inconsistent_edges'] <= 10:
            print("  Inconsistent edges (vertex pairs):")
            for edge in result['inconsistent_edges']:
                print(f"    {edge[0]} -- {edge[1]}")
        else:
            print("  (Too many to list, showing first 10)")
            for edge in result['inconsistent_edges'][:10]:
                print(f"    {edge[0]} -- {edge[1]}")

    if result['n_nonmanifold_edges'] > 0:
        print(f"\n  WARNING: {result['n_nonmanifold_edges']} non-manifold edges detected")
        if result['n_nonmanifold_edges'] <= 10:
            print("  Non-manifold edges (vertex pairs):")
            for edge in result['nonmanifold_edges']:
                print(f"    {edge[0]} -- {edge[1]}")

    print("\n" + "=" * 60)
    if result['consistent']:
        print("RESULT: PASS - Mesh topology is consistent")
    else:
        print("RESULT: FAIL - Mesh has topology issues")
    print("=" * 60)

    return result['consistent']


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_file = sys.argv[1]

    print(f"Reading {input_file}...")
    mesh = read_mesh(input_file)

    if len(mesh.triangles) == 0 and len(mesh.quads) == 0:
        print("Error: Mesh has no faces")
        sys.exit(1)

    result = check_mesh_topology(mesh)
    success = print_report(mesh, result)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
