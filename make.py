import graph

g = graph.BinaryERGraph(50, 0.1)
g.print_adjacency_matrix()
print(g.bfs_from_node(0))
g.print_path(g.generate_random_path(4))
