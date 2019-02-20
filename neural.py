import numpy as np
import kohonen as koh


n = koh.Neural_network()
n.normalize()
n.self_learning(m=0.01, count=30 )
