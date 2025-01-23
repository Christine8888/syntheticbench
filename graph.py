import numpy as np
from abc import ABC, abstractmethod

# GRAPH TRAVERSAL TASKS
# 1. BFS from a node, validatable
# 2. path following, validatable

class Graph(ABC):
    @abstractmethod
    def __init__(self, N):
        pass

    def print_adjacency_matrix(self):
        """
        print adjacency matrix in readable format
        """
        adj_string = "     " + " ".join(f"{j:2d}" for j in range(self.N)) + "\n"
        adj_string += "    " + "---"*self.N + "\n"
        for i in range(self.N):
            row_str = "  ".join(f"{int(self.adj_matrix[i][j])}" for j in range(self.N))
            adj_string += f"{i:3d}| {row_str}\n"
        
        return adj_string
        
    def bfs_from_node(self, start_node):
        """
        perform BFS from 'start_node' and return the maximum distance to any reachable node.
        """
        visited = [False] * self.N
        distance = [-1] * self.N  # distance array to track depth
        
        visited[start_node] = True
        distance[start_node] = 0
        queue = [start_node]
        
        while queue:
            current = queue.pop(0)
            neighbors = self.nodes[self.adj_matrix[current] == 1.0]
            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    distance[neighbor] = distance[current] + 1
                    queue.append(neighbor)
        
        print(distance)
        if max(distance) == -1: return 0
        else: return max(distance)

    def print_path(self, path):
        """
        print path in readable format
        """
        return " -> ".join(f"{i:2d}" for i in path)
    
    def str_to_path(self, path_str):
        """
        convert string of node indices to path
        """
        path = []
        for i in path_str.split(" -> "):
            if i != " ":
                try:
                    path.append(int(i))
                except:
                    print(f"Error converting {i} to int")
                    return path
        return path

    def generate_random_path(self, start_node, path_length=5):
        """
        generate random path of up to length N starting from start_node
        """
        path = [start_node]
        current = start_node
        
        for i in range(path_length - 1):
            neighbors = self.nodes[self.adj_matrix[current] == 1.0]
            try:
                next_node = np.random.choice(neighbors)
            except:
                raise ValueError("No neighbors found")
            path.append(next_node)
            current = next_node
        
        return path

    def is_valid_path(self, path):
        """
        check validity of a path through the graph
        """
        if isinstance(path, str):
            path = self.str_to_path(path)
        
        if len(path) < 2:
            return True
        
        for i in range(len(path) - 1):
            n1, n2 = path[i], path[i + 1]
            if self.adj_matrix[n1][n2] == 0.0:  
                return False
        
        return True

class BinaryERGraph(Graph):
    def __init__(self, N, p):
        """ undirected Erdos-Renyi graph G(n, p).
        N: number of nodes
        p: probability of an edge between any two nodes
        """
        self.nodes = np.array([i for i in range(N)])
        self.N = N
        self.p = p
        upper_tri = np.random.choice([0, 1], size=(N, N), p=[1-p, p])
        np.fill_diagonal(upper_tri, 0)
        lower_indices = np.tril_indices(N, -1)
        upper_tri[lower_indices] = 0
        
        self.adj_matrix = (upper_tri + upper_tri.T).astype(np.float32)
            