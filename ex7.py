# Amnon Ophir 302445804, Ross Bolotin 310918610

import numpy as np
import random
from scipy.stats import multivariate_normal as mvn


def random_number(start, stop):
    number = random.uniform(start, stop)
    while number == start or number == stop:
        number = random.uniform(start, stop)
    return number


N = 2000  # number of samples
c1 = 0.5
c2 = 0.5
cov1 = np.array([[1, 0], [0, 2]])
cov2 = np.array([[2, 0], [0, 0.5]])
mean1 = np.array([1, 1]).transpose()
mean2 = np.array([3, 3]).transpose()

np.random.seed(777)

#  section A:

x1 = np.random.multivariate_normal(mean1, cov1, int(c1 * N)).T
x2 = np.random.multivariate_normal(mean2, cov2, int(c2 * N)).T
x = np.concatenate((x1,x2), axis=1).T

# theta = sigma1, sigma2, mus, c_learn
# theta initialization:
c_learn = np.zeros(2)
c_learn[0] = random_number(0, 1)
c_learn[1] = 1 - c_learn[0]
sigma1 = np.zeros((2, 2))
sigma1[0][0] = random_number(0, 5)
sigma1[1][1] = random_number(0, 5)
sigma2 = np.zeros((2, 2))
sigma2[0][0] = random_number(0, 5)
sigma2[1][1] = random_number(0, 5)
mus = np.zeros((2, 2))
mus[0][0] = random_number(0, 5)
mus[1][0] = random_number(0, 5)
mus[0][1] = random_number(0, 5)
mus[1][1] = random_number(0, 5)

for iter in range(100):
    # E step:
    alpha = np.zeros((2, N))
    for l in range(2):
        for j in range(N):
            if l == 0:
                sigma = sigma1
            else:
                sigma = sigma2
            alpha[l, j] = c_learn[l] * mvn(mus[l], sigma).pdf(x[j])
    alpha /= alpha.sum(0)

    # M step:
    # obtaining c_learn:
    c_learn = np.zeros(2)
    for l in range(2):
        for j in range(N):
            c_learn[l] += alpha[l, j]
    c_learn /= N
    # obtaining mu:
    mus = np.zeros((2, 2))
    for l in range(2):
        for j in range(N):
            mus[l] += alpha[l, j] * x[j]
        mus[l] /= alpha[l, :].sum()
    # obtaining sigma:
    sigma1 = np.zeros((2, 2))
    sigma2 = np.zeros((2, 2))
    for l in range(2):
        for i in range(2):
            for j in range(N):
                d = x[j][i] - mus[l][i]
                if l == 0:
                    sigma1[i][i] += alpha[l, j] * d ** 2
                else:
                    sigma2[i][i] += alpha[l, j] * d ** 2
            if l == 0:
                sigma1[i][i] /= alpha[l, :].sum()
            else:
                sigma2[i][i] /= alpha[l, :].sum()

    # print estimated parameters:
    if iter == 1 or iter == 9 or iter == 99:
        print("EM estimated parameters after ", iter + 1, "iterations are:")
        print("c1 = ", c_learn[0])
        print("c2 = ", c_learn[1])
        print("sigma1 = ", sigma1)
        print("sigma2 = ", sigma2)
        print("mu1 = ", mus[0])
        print("mu2 = ", mus[1])


#  section B:

means = {0: x[0], 1: x[1]}  # means initialization

for i in range(100):
    classifications = {0: [], 1: []}
    # classify by min distance:
    for point in x:
        distances = [np.linalg.norm(point - means[index]) for index in means]
        classification = distances.index(min(distances))
        classifications[classification].append(point)
    # set new means by average:
    for classification in classifications:
        means[classification] = np.average(classifications[classification], axis=0)
    # print means:
    if i == 1 or i == 9 or i == 99:
        print("K-Means estimated means after ", i + 1, "iterations are: ", means)
