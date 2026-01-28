"""
VTK Writer for UGRID Data

Converts UGRID data to VTK Unstructured Grid format (.vtu or .vtk)
Supports both ASCII and XML formats.
"""

import numpy as np
from typing import Dict, List, Optional, Union


class VTKWriter:
    """Write unstructured grid data to VTK format"""
    
    # VTK cell type IDs
    VTK_VERTEX = 1
    VTK_LINE = 3
    VTK_TRIANGLE = 5
    VTK_QUAD = 9
    VTK_TETRA = 10
    VTK_HEXAHEDRON = 12
    VTK_WEDGE = 13
    VTK_PYRAMID = 14
    
    def __init__(self):
        self.points = None
        self.cells = None
        self.cell_types = None
        self.point_data = {}
        self.cell_data = {}
        
    def set_points(self, points: np.ndarray):
        """
        Set grid points/vertices
        
        Args:
            points: Nx3 array of point coordinates (x, y, z)
        """
        if points.shape[1] == 2:
            # If 2D, add z=0 coordinate
            points = np.column_stack([points, np.zeros(len(points))])
        self.points = points.astype(np.float64)
        
    def set_cells(self, cells: List[np.ndarray], cell_types: Optional[List[int]] = None):
        """
        Set grid cells
        
        Args:
            cells: List of arrays, where each array contains vertex indices for one cell
            cell_types: List of VTK cell type IDs. If None, will auto-detect from cell sizes
        """
        self.cells = cells
        
        if cell_types is None:
            # Auto-detect cell types based on number of vertices
            self.cell_types = []
            for cell in cells:
                n_verts = len(cell)
                if n_verts == 1:
                    self.cell_types.append(self.VTK_VERTEX)
                elif n_verts == 2:
                    self.cell_types.append(self.VTK_LINE)
                elif n_verts == 3:
                    self.cell_types.append(self.VTK_TRIANGLE)
                elif n_verts == 4:
                    self.cell_types.append(self.VTK_QUAD)  # Could also be tetra
                elif n_verts == 8:
                    self.cell_types.append(self.VTK_HEXAHEDRON)
                else:
                    raise ValueError(f"Cannot auto-detect cell type for {n_verts} vertices")
        else:
            self.cell_types = cell_types
            
    def add_point_data(self, name: str, data: np.ndarray):
        """
        Add scalar or vector data associated with points
        
        Args:
            name: Name of the data field
            data: Array of data values (length must match number of points)
        """
        if len(data) != len(self.points):
            raise ValueError(f"Point data length {len(data)} doesn't match number of points {len(self.points)}")
        self.point_data[name] = data
        
    def add_cell_data(self, name: str, data: np.ndarray):
        """
        Add scalar or vector data associated with cells
        
        Args:
            name: Name of the data field
            data: Array of data values (length must match number of cells)
        """
        if len(data) != len(self.cells):
            raise ValueError(f"Cell data length {len(data)} doesn't match number of cells {len(self.cells)}")
        self.cell_data[name] = data
        
    def write_vtk_legacy(self, filename: str, title: str = "UGRID Mesh"):
        """
        Write to legacy VTK ASCII format (.vtk)
        
        Args:
            filename: Output filename
            title: Title for the VTK file header
        """
        with open(filename, 'w') as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"{title}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            
            # Points
            n_points = len(self.points)
            f.write(f"POINTS {n_points} double\n")
            for point in self.points:
                f.write(f"{point[0]:.10e} {point[1]:.10e} {point[2]:.10e}\n")
            
            # Cells
            n_cells = len(self.cells)
            cell_list_size = sum(len(cell) + 1 for cell in self.cells)
            f.write(f"\nCELLS {n_cells} {cell_list_size}\n")
            for cell in self.cells:
                f.write(f"{len(cell)}")
                for idx in cell:
                    f.write(f" {idx}")
                f.write("\n")
            
            # Cell types
            f.write(f"\nCELL_TYPES {n_cells}\n")
            for cell_type in self.cell_types:
                f.write(f"{cell_type}\n")
            
            # Point data
            if self.point_data:
                f.write(f"\nPOINT_DATA {n_points}\n")
                for name, data in self.point_data.items():
                    self._write_data_array(f, name, data, "point")
            
            # Cell data
            if self.cell_data:
                f.write(f"\nCELL_DATA {n_cells}\n")
                for name, data in self.cell_data.items():
                    self._write_data_array(f, name, data, "cell")
                    
    def _write_data_array(self, f, name: str, data: np.ndarray, data_type: str):
        """Helper to write data arrays in legacy format"""
        if data.ndim == 1:
            # Scalar data
            f.write(f"SCALARS {name} double\n")
            f.write("LOOKUP_TABLE default\n")
            for value in data:
                f.write(f"{value:.10e}\n")
        elif data.ndim == 2:
            # Vector data
            n_components = data.shape[1]
            if n_components == 3:
                f.write(f"VECTORS {name} double\n")
            else:
                f.write(f"SCALARS {name} double {n_components}\n")
                f.write("LOOKUP_TABLE default\n")
            for row in data:
                f.write(" ".join(f"{v:.10e}" for v in row))
                f.write("\n")
                
    def write_vtu_xml(self, filename: str):
        """
        Write to VTK XML format (.vtu)
        Uses ASCII encoding for simplicity
        
        Args:
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian">\n')
            f.write('  <UnstructuredGrid>\n')
            
            n_points = len(self.points)
            n_cells = len(self.cells)
            f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')
            
            # Points
            f.write('      <Points>\n')
            f.write('        <DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
            for point in self.points:
                f.write(f'          {point[0]:.10e} {point[1]:.10e} {point[2]:.10e}\n')
            f.write('        </DataArray>\n')
            f.write('      </Points>\n')
            
            # Cells
            f.write('      <Cells>\n')
            
            # Connectivity
            f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
            f.write('          ')
            for cell in self.cells:
                f.write(' '.join(str(idx) for idx in cell))
                f.write(' ')
            f.write('\n        </DataArray>\n')
            
            # Offsets
            f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
            f.write('          ')
            offset = 0
            for cell in self.cells:
                offset += len(cell)
                f.write(f'{offset} ')
            f.write('\n        </DataArray>\n')
            
            # Types
            f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
            f.write('          ')
            for cell_type in self.cell_types:
                f.write(f'{cell_type} ')
            f.write('\n        </DataArray>\n')
            
            f.write('      </Cells>\n')
            
            # Point data
            if self.point_data:
                f.write('      <PointData>\n')
                for name, data in self.point_data.items():
                    self._write_xml_data_array(f, name, data)
                f.write('      </PointData>\n')
            
            # Cell data
            if self.cell_data:
                f.write('      <CellData>\n')
                for name, data in self.cell_data.items():
                    self._write_xml_data_array(f, name, data)
                f.write('      </CellData>\n')
            
            f.write('    </Piece>\n')
            f.write('  </UnstructuredGrid>\n')
            f.write('</VTKFile>\n')
            
    def _write_xml_data_array(self, f, name: str, data: np.ndarray):
        """Helper to write data arrays in XML format"""
        if data.ndim == 1:
            n_components = 1
        else:
            n_components = data.shape[1]
            
        f.write(f'        <DataArray type="Float64" Name="{name}" NumberOfComponents="{n_components}" format="ascii">\n')
        f.write('          ')
        
        if data.ndim == 1:
            for value in data:
                f.write(f'{value:.10e} ')
        else:
            for row in data:
                f.write(' '.join(f'{v:.10e}' for v in row))
                f.write(' ')
        
        f.write('\n        </DataArray>\n')


def write_ugrid_to_vtk(ugrid_data: Dict, output_file: str, format: str = 'xml'):
    """
    Convenience function to write UGRID data to VTK
    
    Args:
        ugrid_data: Dictionary containing UGRID data with keys:
            - 'points' or 'nodes': Nx3 array of coordinates
            - 'cells' or 'elements': List of cell connectivity arrays
            - 'cell_types' (optional): List of VTK cell type IDs
            - 'point_data' (optional): Dict of point-associated data
            - 'cell_data' (optional): Dict of cell-associated data
        output_file: Output filename
        format: 'xml' for .vtu or 'legacy' for .vtk
    
    Example:
        ugrid_data = {
            'points': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
            'cells': [np.array([0, 1, 2])],
            'point_data': {'pressure': np.array([1.0, 2.0, 3.0])}
        }
        write_ugrid_to_vtk(ugrid_data, 'output.vtu')
    """
    writer = VTKWriter()
    
    # Set points
    points = ugrid_data.get('points')
    if points is None:
        points = ugrid_data.get('nodes')
    if points is None:
        raise ValueError("UGRID data must contain 'points' or 'nodes'")
    writer.set_points(points)
    
    # Set cells
    cells = ugrid_data.get('cells')
    if cells is None:
        cells = ugrid_data.get('elements')
    if cells is None:
        raise ValueError("UGRID data must contain 'cells' or 'elements'")
    cell_types = ugrid_data.get('cell_types')
    writer.set_cells(cells, cell_types)
    
    # Add point data
    if 'point_data' in ugrid_data:
        for name, data in ugrid_data['point_data'].items():
            writer.add_point_data(name, data)
    
    # Add cell data
    if 'cell_data' in ugrid_data:
        for name, data in ugrid_data['cell_data'].items():
            writer.add_cell_data(name, data)
    
    # Write file
    if format.lower() == 'xml':
        if not output_file.endswith('.vtu'):
            output_file += '.vtu'
        writer.write_vtu_xml(output_file)
    else:
        if not output_file.endswith('.vtk'):
            output_file += '.vtk'
        writer.write_vtk_legacy(output_file)
    
    print(f"Successfully wrote {output_file}")


if __name__ == "__main__":
    # Example usage
    print("VTK Writer for UGRID Data")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from vtk_writer import VTKWriter, write_ugrid_to_vtk
    
    # Method 1: Using the convenience function
    ugrid_data = {
        'points': points_array,  # Nx3 numpy array
        'cells': cells_list,     # List of numpy arrays
        'point_data': {          # Optional
            'pressure': pressure_array,
            'velocity': velocity_array
        }
    }
    write_ugrid_to_vtk(ugrid_data, 'output.vtu', format='xml')
    
    # Method 2: Using the writer class directly
    writer = VTKWriter()
    writer.set_points(points_array)
    writer.set_cells(cells_list)
    writer.add_point_data('pressure', pressure_array)
    writer.write_vtu_xml('output.vtu')
    # or writer.write_vtk_legacy('output.vtk')
    """)
