import numpy as np
from random import random
from numba import njit
import random as rand
import matplotlib.pyplot as plt
# This enforces the use of Tensorflow 1.x instead of the default Tensorflow 2.x
#%tensorflow_version 1.x
import tensorflow as tf
import tf_slim as slim
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #stop displaying tensorflow warnings
import numpy as np
import os
import time
import random
import matplotlib.pyplot as plt

#For plotting purposes
from matplotlib import rcParams

rcParams['axes.labelsize']  = 20
rcParams['font.serif']      = ['Computer Modern']
rcParams['font.size']       = 10
rcParams['legend.fontsize'] = 20
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from typing import Any, Optional, Union, Text, Sequence, Tuple, List