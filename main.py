import numpy as np
from module import MLP
from pathlib import Path
import os

path = os.path.dirname(os.path.abspath("__file__"))
file = str(path) + "\\iris3.csv"

data = np.genfromtxt(file, skip_header=True, delimiter=',')

slp1 = MLP(data,1000,0.6) #data, epoch, k-fold, learning rate 0.1

#uncomment one only
slp1.run()
