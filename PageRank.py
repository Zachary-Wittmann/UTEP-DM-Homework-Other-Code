# Created and formatted by Zachary Wittmann
# Version 1.8

import numpy as np
from numpy import array
import matplotlib.pyplot as plt

# Threshold for page rank updating
threshold = 0.00000001

# Damping factor for probability of continued surfing
beta = 0.8

# Allow for teleportation even if not a spider trap
teleportation = True

# # Normal Adjacency Matrix (3x3)
# A = array([[1, 1, 0],
#            [1, 0, 1],
#            [0, 1, 0]], dtype=float)

# # Spider Trap Adjacency Matrix (3x3)
# A = array([[1, 1, 0],
#            [1, 0, 0],
#            [1, 1, 1]], dtype=float)

# Dead-End Adjacency Matrix at index 2 (3x3)
A = array([[1, 1, 0],
           [0, 1, 0],
           [1, 0, 0]], dtype=float)

# # Normal Adjacency Matrix (3x3) (num > 1)
# A = array([[4, 4, 0],
#            [4, 0, 4],
#            [0, 4, 0]], dtype=float)

# # Spider Trap Adjacency Matrix (3x3) (num > 1)
# A = array([[4, 2, 0],
#            [5, 0, 0],
#            [8, 9, 3]], dtype=float)

# # Normal Adjacency Matrix (4x4)
# A = array([[1, 1, 0, 0],
#            [0, 1, 0, 1],
#            [1, 0, 1, 0],
#            [0, 0, 1, 1]], dtype=float)

# # Spider Trap Adjacency Matrix (4x4)
# A = array([[1, 0, 0, 1],
#            [0, 1, 1, 0],
#            [1, 0, 1, 0],
#            [0, 0, 1, 1]], dtype=float)

# # Dead-End Adjacency Matrix at index 3 (4x4)
# A = array([[1, 0, 0, 0],
#            [0, 1, 1, 0],
#            [1, 0, 1, 0],
#            [0, 0, 1, 0]], dtype=float)


def loss(prev, curr):
    diff = curr - prev
    diffSquared = diff ** 2
    meanDiff = diffSquared.mean()
    return meanDiff


def spiderTrap(adjMat):
    # Find Spider Trap from Adjacency Matrix
    numOfZeros = list(np.sum(adjMat == 0, axis=0))

    if (max(numOfZeros) != len(adjMat) - 1):
        return False

    sTIndex = numOfZeros.index(len(adjMat) - 1)
    print(sTIndex)
    hasSpiderTrap = (0.0 != ([r[i] for i, r in enumerate(adjMat)][sTIndex]))
    return hasSpiderTrap


def deadEnd(adjMat):
    # Replace column of dead end to 1.0, then change the value going to
    # the dead-end into an infinitesimally small value
    deadEnd = list(map(sum, zip(*adjMat)))
    if 0.0 not in deadEnd:
        return False
    dEIndex = list(map(sum, zip(*adjMat))).index(0.0)
    adjMat[:, dEIndex] = 1.0
    dEPos = list(adjMat[dEIndex]).index(1.0)
    adjMat[dEIndex, dEPos] = threshold
    return True


def colStoMatrix(adjMat):
    s = [np.sum(adjMat[:, i]) for i in range(0, len(adjMat))]
    M = adjMat
    M = np.array([M[:, j] / s[j] for j in range(len(s))]).T
    return M


def rankVectorNorm(adjMat, colStoMat, printProc=False, plotProc=False, step=1):
    r = np.ones([len(adjMat), 1])/len(adjMat)
    rPrev = r
    itr = 0

    print(f"Initial Iteration:\n{r}\n")

    if (plotProc):
        plt.figure()

    for i in range(0, 1000):
        r = np.matmul(colStoMat, rPrev)

        if (printProc and i % step == 0):
            print(f"Iteration: {i}\n{r}\n")

        if (plotProc and i % step == 0):
            plt.title("No Teleporation")
            plt.scatter(loss(rPrev, r), i)
            plt.xlabel("Loss")
            plt.ylabel("Iteration")

        diff = sum(abs(r - rPrev))

        if (diff <= threshold):
            itr = i
            break
        rPrev = r

    print(f"Final Iteration ({itr}):\n{r}")


def rankVectorTp(adjMat, colStoMat, printProc=False, plotProc=False, step=1):
    r = np.ones([len(adjMat), 1])/len(adjMat)
    rPrev = r
    itr = 0

    print(f"Initial Iteration:\n{r}\n")

    # Vector for teleportation
    const = (1.0 - beta) * r

    if (plotProc):
        plt.figure()

    for i in range(0, 1000):
        r = beta * np.matmul(colStoMat, rPrev) + const

        if (printProc and i % step == 0):
            print(f"Iteration: {i}\n{r}\n")

        if (plotProc and i % step == 0):
            plt.title(f"Teleporation with Beta: {beta}")
            plt.scatter(loss(rPrev, r), i)
            plt.xlabel("Loss")
            plt.ylabel("Iteration")

        diff = sum(abs(r - rPrev))

        if (diff <= threshold):
            itr = i
            break
        rPrev = r

    print(f"Final Iteration ({itr}):\n{r}")


if __name__ == "__main__":

    # print(spiderTrap(A))
    # print(colStoMatrix(A))

    deadEnd(A)
    M = colStoMatrix(A)

    if (teleportation):
        rankVectorTp(A, M, printProc=True, plotProc=True, step=10)
    else:
        rankVectorNorm(A, M, printProc=True, plotProc=True, step=10)
