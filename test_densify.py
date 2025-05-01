import numpy as np

from continuous_tools import densify_polygon_edges

square = np.array([[0,0], [0,100], [100,100], [100,0]])
dense = densify_polygon_edges(square, spacing=10)
print(dense.shape)  # Expect ~41 points