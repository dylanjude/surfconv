# UGRID to VTK/STL Converter - Quick Reference

## Files Included

1. **vtk_writer.py** - VTK format writer module
2. **stl_writer.py** - STL format writer module  
3. **ugrid2vtk.py** - Main converter script (handles both VTK and STL)
4. **ugrid2stl.py** - Dedicated STL converter

## Quick Usage

### Convert to VTK (recommended)
```bash
# XML format (modern, recommended)
python ugrid2vtk.py mesh.ugrid

# Legacy format
python ugrid2vtk.py mesh.ugrid mesh.vtk --vtk
```

### Convert to STL
```bash
# Binary STL (compact)
python ugrid2vtk.py mesh.ugrid mesh.stl --stl

# ASCII STL (human-readable)
python ugrid2vtk.py mesh.ugrid mesh.stl --stl-ascii

# Or use the dedicated converter
python ugrid2stl.py mesh.ugrid mesh.stl
```

## Format Comparison

| Format | Extension | Size | Visualization | Best For |
|--------|-----------|------|---------------|----------|
| VTK XML | .vtu | Medium | ParaView, VisIt | General viz, keeps quads |
| VTK Legacy | .vtk | Large | ParaView, VisIt | Compatibility |
| Binary STL | .stl | Small | CAD software | 3D printing, triangles only |
| ASCII STL | .stl | Large | CAD software | Human-readable |

## Features

### VTK Output
- Preserves both triangles and quads separately
- Can attach field data (pressure, velocity, etc.)
- Better for scientific visualization
- Recommended for ParaView/VisIt

### STL Output
- Converts all quads to triangles
- Computes surface normals automatically
- Better for CAD/3D printing
- Simpler format

## Integration with Your Code

Your existing UGRID reader:
```python
with open(fin, "r") as f:
    nv,n3f,n4f,n4,n5,n6,n8 = [int(xx) for xx in f.readline().split()]
    x = np.zeros((nv,3),'d')
    conn3 = np.zeros((n3f,3),'i')
    conn4 = np.zeros((n4f,4),'i')    
    for i in range(nv):
        x[i,:] = [float(xx) for xx in f.readline().split()]
    for i in range(n3f):
        conn3[i,:] = [int(xx) for xx in f.readline().split()]
    for i in range(n4f):
        conn4[i,:] = [int(xx) for xx in f.readline().split()]
```

### To write VTK:
```python
from vtk_writer import VTKWriter

# Prepare cells (adjust for 1-based indexing if needed)
cells = []
cell_types = []

for tri in conn3:
    cells.append(tri - 1)  # Convert to 0-based if needed
    cell_types.append(5)   # VTK_TRIANGLE

for quad in conn4:
    cells.append(quad - 1) # Convert to 0-based if needed
    cell_types.append(9)   # VTK_QUAD

# Write
writer = VTKWriter()
writer.set_points(x)
writer.set_cells(cells, cell_types)
writer.write_vtu_xml('output.vtu')
```

### To write STL:
```python
from stl_writer import write_stl, quads_to_triangles

# Combine triangles and quads (convert quads to triangles)
all_triangles = conn3.copy()
if len(conn4) > 0:
    quad_triangles = quads_to_triangles(conn4)
    all_triangles = np.vstack([all_triangles, quad_triangles])

# Adjust indexing if needed
if all_triangles.min() == 1:
    all_triangles = all_triangles - 1

# Write
write_stl(x, all_triangles, 'output.stl', format='binary')
```

## Notes

- **Indexing**: The converter automatically detects and handles 1-based vs 0-based indexing
- **Quads in STL**: Quads are automatically split into 2 triangles (vertices [0,1,2] and [0,2,3])
- **Normals**: STL normals are computed automatically using the right-hand rule
- **2D meshes**: If your mesh is 2D (z=0), the VTK writer handles this automatically

## Visualization Software

- **ParaView** (free): https://www.paraview.org/
- **VisIt** (free): https://visit.llnl.gov/
- **Blender** (free): Can import STL files
- **MeshLab** (free): Good for STL viewing/editing

## Command Line Examples

```bash
# Simple conversion to default format
python ugrid2vtk.py wing.ugrid

# Multiple outputs
python ugrid2vtk.py wing.ugrid wing.vtu --vtu
python ugrid2vtk.py wing.ugrid wing.stl --stl

# ASCII formats for debugging
python ugrid2vtk.py wing.ugrid wing.vtk --vtk
python ugrid2vtk.py wing.ugrid wing.stl --stl-ascii
```
