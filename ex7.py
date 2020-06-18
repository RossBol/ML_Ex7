# Amnon Ophir 302445804, Ross Bolotin 310918610

import numpy as np
import matplotlib.pyplot as plt


N = 2000  # number of samples
c1 = 0.5
c2 = 0.5
cov1 = np.array([[1, 0], [0, 2]])
cov2 = np.array([[2, 0], [0, 0.5]])
mean1 = np.array([1, 1]).transpose()
mean2 = np.array([3, 3]).transpose()

np.random.seed(777)

x1, y1 = np.random.multivariate_normal(mean1, cov1, int(N/2)).T
x2, y2 = np.random.multivariate_normal(mean2, cov2, int(N/2)).T

plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'x')
plt.axis('equal')
plt.show()
