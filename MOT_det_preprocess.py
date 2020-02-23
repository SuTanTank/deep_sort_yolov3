import numpy as np

det = np.loadtxt("det-MOT17-03.txt", delimiter=',')
det[:, 1] = 1
np.savetxt("det-MOT17-03-p.txt", det, delimiter=',')
