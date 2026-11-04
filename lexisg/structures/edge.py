import numpy as np

class Edge: 
    def __init__(
        self, 
        source: int,
        target: int,
        edge_type: str,
        boundary_pairs: dict = None,
        relative_transform: np.ndarray = None,
    ):
        self.source = source 
        self.target = target
        self.relative_transform = relative_transform
        self.edge_type = edge_type
        self.boundary_pairs = boundary_pairs
    
    def __repr__(self) -> str:
        return (
            f"Edge(source={self.source}) "
            f"target={self.target} "
            f"transform_shape={'set' if self.relative_transform is not None else 'None'})"
        )
