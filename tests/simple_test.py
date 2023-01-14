import pyastar2d
import numpy as np 

weights = np.ones((5,5), dtype=np.float32)
start=(0,0)
goal=(4,4)

path = pyastar2d.astar_path(weights,start,goal)
