#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from utils.visualize import plot_scene

for fn in sys.argv[1:]:
    d = np.load(fn)
    plot_scene(d)
    plt.show()
