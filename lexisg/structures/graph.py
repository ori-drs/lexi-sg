from lexisg.structures.node import Node, RoomNode
from lexisg.structures.edge import Edge

class Graph:
    def  __init__(self):
        self.room_nodes = {}
        self.room_to_room_edge = {}
        self.adj = {}
    
    def add_node(self, node: Node): 
        if isinstance(node, RoomNode): 
            self.room_nodes[node.idx] = node 
        else: 
            raise TypeError(f"Unsupported node type: {type(node)}")
    
    def add_edge(self, edge: Edge): 
        src = edge.source 
        tgt = edge.target
        key = (src, tgt)
        if key not in self.room_to_room_edge:
            self.room_to_room_edge[key] = []
        self.room_to_room_edge[key].append(edge)

        # Adjacency matrix for fast look up 
        if src not in self.adj: self.adj[src] = []
        if tgt not in self.adj: self.adj[tgt] = []
        
        if tgt not in self.adj[src]:
            self.adj[src].append(tgt)
        
        if src not in self.adj[tgt]: 
            self.adj[tgt].append(src)
