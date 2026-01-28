"""
STL Writer for Surface Mesh Data

Converts triangular surface mesh data to STL format.
Supports both ASCII and binary STL formats.
"""

import numpy as np
from typing import Optional


class STLWriter:
    """Write triangular surface meshes to STL format"""
    
    def __init__(self):
        self.vertices = None
        self.triangles = None
        
    def set_mesh(self, vertices: np.ndarray, triangles: np.ndarray):
        """
        Set the mesh data
        
        Args:
            vertices: Nx3 array of vertex coordinates
            triangles: Mx3 array of triangle connectivity (vertex indices)
        """
        self.vertices = vertices.astype(np.float64)
        self.triangles = triangles.astype(np.int32)
        
    def _compute_normal(self, v0, v1, v2):
        """Compute unit normal vector for a triangle using right-hand rule"""
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        else:
            normal = np.array([0.0, 0.0, 0.0])
        return normal
        
    def write_ascii(self, filename: str, solid_name: str = "mesh"):
        """
        Write ASCII STL format
        
        Args:
            filename: Output filename
            solid_name: Name of the solid (appears in STL header)
        """
        with open(filename, 'w') as f:
            f.write(f"solid {solid_name}\n")
            
            for tri in self.triangles:
                # Get triangle vertices
                v0 = self.vertices[tri[0]]
                v1 = self.vertices[tri[1]]
                v2 = self.vertices[tri[2]]
                
                # Compute normal
                normal = self._compute_normal(v0, v1, v2)
                
                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write(f"    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write(f"    endloop\n")
                f.write(f"  endfacet\n")
            
            f.write(f"endsolid {solid_name}\n")
            
    def write_binary(self, filename: str, header: str = "Binary STL"):
        """
        Write binary STL format
        
        Args:
            filename: Output filename
            header: 80-character header string
        """
        import struct
        
        # Prepare header (must be exactly 80 bytes)
        header_bytes = header.encode('ascii')[:80].ljust(80, b'\0')
        
        n_triangles = len(self.triangles)
        
        with open(filename, 'wb') as f:
            # Write header
            f.write(header_bytes)
            
            # Write number of triangles
            f.write(struct.pack('<I', n_triangles))
            
            # Write each triangle
            for tri in self.triangles:
                # Get triangle vertices
                v0 = self.vertices[tri[0]]
                v1 = self.vertices[tri[1]]
                v2 = self.vertices[tri[2]]
                
                # Compute normal
                normal = self._compute_normal(v0, v1, v2)
                
                # Write normal (3 floats)
                f.write(struct.pack('<fff', *normal))
                
                # Write vertices (9 floats)
                f.write(struct.pack('<fff', *v0))
                f.write(struct.pack('<fff', *v1))
                f.write(struct.pack('<fff', *v2))
                
                # Write attribute byte count (uint16, typically 0)
                f.write(struct.pack('<H', 0))


def write_stl(vertices: np.ndarray, triangles: np.ndarray, 
              filename: str, format: str = 'binary', **kwargs):
    """
    Convenience function to write mesh to STL
    
    Args:
        vertices: Nx3 array of vertex coordinates
        triangles: Mx3 array of triangle connectivity
        filename: Output filename
        format: 'binary' (default) or 'ascii'
        **kwargs: Additional arguments passed to write methods
            - solid_name: for ASCII format
            - header: for binary format
    
    Example:
        vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
        triangles = np.array([[0,1,2]])
        write_stl(vertices, triangles, 'output.stl')
    """
    writer = STLWriter()
    writer.set_mesh(vertices, triangles)
    
    if format.lower() == 'ascii':
        solid_name = kwargs.get('solid_name', 'mesh')
        writer.write_ascii(filename, solid_name)
    else:
        header = kwargs.get('header', 'Binary STL')
        writer.write_binary(filename, header)
    
    print(f"Successfully wrote {filename} ({format} format)")


def quads_to_triangles(quads: np.ndarray) -> np.ndarray:
    """
    Convert quad connectivity to triangles
    
    Args:
        quads: Nx4 array of quad connectivity
        
    Returns:
        Mx3 array of triangle connectivity (M = 2*N)
        
    Note:
        Splits each quad into two triangles using vertices [0,1,2] and [0,2,3]
    """
    n_quads = len(quads)
    triangles = np.zeros((n_quads * 2, 3), dtype=np.int32)
    
    for i, quad in enumerate(quads):
        # First triangle: vertices 0, 1, 2
        triangles[2*i] = [quad[0], quad[1], quad[2]]
        # Second triangle: vertices 0, 2, 3
        triangles[2*i + 1] = [quad[0], quad[2], quad[3]]
    
    return triangles


if __name__ == "__main__":
    print("STL Writer for Surface Meshes")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from stl_writer import write_stl, STLWriter
    
    # Method 1: Using the convenience function
    write_stl(vertices, triangles, 'output.stl', format='binary')
    write_stl(vertices, triangles, 'output.stl', format='ascii', solid_name='my_mesh')
    
    # Method 2: Using the writer class directly
    writer = STLWriter()
    writer.set_mesh(vertices, triangles)
    writer.write_binary('output.stl')
    # or writer.write_ascii('output.stl', solid_name='my_mesh')
    
    # Converting quads to triangles
    from stl_writer import quads_to_triangles
    triangles = quads_to_triangles(quad_connectivity)
    """)
