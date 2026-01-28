# SurfConv

Surface mesh converter supporting UGRID, VTK, STL, and Pointwise FACET formats.

## Supported Formats

| Format | Extension | Read | Write | Notes |
|--------|-----------|------|-------|-------|
| UGRID | `.ugrid` | Yes | Yes | Triangles and quads |
| VTK Legacy | `.vtk` | Yes | Yes | ASCII unstructured grid |
| VTK XML | `.vtu` | Yes | Yes | XML unstructured grid |
| STL Binary | `.stl` | Yes | Yes | Triangles only |
| STL ASCII | `.stl` | Yes | Yes | Triangles only |
| FACET | `.facet` | Yes | Yes | Pointwise format |

## Requirements

- Python 3.6+
- NumPy

## Usage

### Unified Converter

Convert between any supported formats:

```bash
# Format auto-detected from extensions
python surfconv.py input.ugrid output.vtu
python surfconv.py mesh.stl mesh.facet
python surfconv.py model.vtu model.ugrid

# Specify output format (generates filename automatically)
python surfconv.py mesh.ugrid --format stl      # Creates mesh.stl
python surfconv.py mesh.ugrid --format stl-ascii

# Override format detection
python surfconv.py input.dat output.dat --input-format ugrid --output-format vtk
```

### Python API

```python
from surfconv import read_mesh, write_mesh, convert, SurfaceMesh

# Simple conversion
convert('input.ugrid', 'output.vtu')

# Read and manipulate mesh
mesh = read_mesh('input.ugrid')
print(f"Vertices: {len(mesh.vertices)}")
print(f"Triangles: {len(mesh.triangles)}")
print(f"Quads: {len(mesh.quads)}")

# Write to different format
write_mesh(mesh, 'output.stl')

# Get all faces as triangles (quads split)
all_tris = mesh.all_triangles()
```

## Notes

- Quad elements are preserved when converting between formats that support them (UGRID, VTK, FACET)
- STL only supports triangles; quads are automatically split into two triangles
- Index conversion between 0-based and 1-based formats is handled automatically
