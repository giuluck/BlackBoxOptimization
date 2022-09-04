import numpy as np

from problems import AckleyProblem

if __name__ == '__main__':
    x = np.array([0, 0])
    y = AckleyProblem((-1, 1), (-1, 1)).evaluate(x)
    print(y)
