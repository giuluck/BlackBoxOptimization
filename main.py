import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from eml.backend import cplex_backend as cplex
from eml.net.embed import encode
from matplotlib import gridspec

from acquisitions.lcb import LowerConfidenceBound
from models import NeuralNetwork, Model
from problems import AckleyProblem

problem = AckleyProblem((-50, 100))
surrogate = NeuralNetwork(units=[8, 8], batch_size=8, epochs=5000, std=Model.log_std)
# surrogate = GaussianProcess()
optimizer = NeuralNetwork(units=[8, 8], batch_size=8, epochs=5000)
acquisition = LowerConfidenceBound(theta=2.0)
iterations = 1
initial_samples = 4
acquisition_samples = 100
time_limit = 30
seed = 0


def draw(xf, yf, xa, ya, it):
    # build linear space for drawing
    space = np.linspace(problem.lower_bounds, problem.upper_bounds, endpoint=True, num=1000)
    mu = surrogate.mean(space)
    sigma = surrogate.std(space)
    acq = acquisition.value(mu, sigma)
    plt.figure(figsize=(16, 9), tight_layout=True)
    gs = gridspec.GridSpec(4, 4)
    # plot true function
    ax = plt.subplot(gs[:2, :2])
    plt.plot(space, problem.evaluate(space))
    plt.scatter(xf[:-1], yf[:-1], color='black', label='points')
    plt.scatter(xf[-1:], yf[-1:], color='red', label='optima')
    plt.title('True Function')
    plt.legend()
    # plot surrogate model
    plt.subplot(gs[:2, 2:], sharex=ax, sharey=ax)
    plt.plot(space, mu)
    plt.fill_between(space[:, 0], mu - sigma, mu + sigma, alpha=0.3)
    plt.scatter(xf[:-1], yf[:-1], color='black', label='points')
    plt.scatter(xf[-1:], yf[-1:], color='red', label='optima')
    plt.vlines(xf[:-1], yf[:-1], surrogate.mean(xf[:-1]), linestyles='--', color='black')
    plt.vlines(xf[-1:], yf[-1:], surrogate.mean(xf[-1:]), linestyles='--', color='red')
    plt.title('Surrogate Function')
    plt.legend()
    # plot acquisition function
    plt.subplot(gs[2:, :2], sharex=ax, sharey=ax)
    plt.plot(space, acq)
    plt.scatter(xa, ya, color='black', alpha=0.5, label='points')
    plt.scatter(xf[-1:], yf[-1:], color='red', label='optima')
    mxf, sxf = surrogate.mean(xf[-1:]), surrogate.std(xf[-1:])
    plt.vlines(xf[-1:], yf[-1:], acquisition.value(mxf, sxf), linestyles='--', color='red')
    plt.title('Acquisition Function')
    plt.legend()
    # plot surrogate acquisition
    plt.subplot(gs[2:, 2:], sharex=ax, sharey=ax)
    plt.plot(space, optimizer.mean(space))
    plt.scatter(xa, ya, color='black', alpha=0.5, label='points')
    plt.scatter(xf[-1:], yf[-1:], color='red', label='optima')
    plt.vlines(xa, ya, optimizer.mean(xa), alpha=0.2, linestyles='--', color='black')
    plt.vlines(xf[-1:], yf[-1:], optimizer.mean(xf[-1:]), linestyles='--', color='red')
    plt.title('Surrogate Acquisition')
    plt.legend()
    # plot upper title
    plt.suptitle(f"Iteration {it}")
    plt.show()


if __name__ == '__main__':
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    x = problem.sample(n=initial_samples)
    y = problem.evaluate(x)
    for i in range(iterations):
        print(f'ITERATION {i + 1}')
        print('     > fitting the surrogate model...')
        surrogate.fit(x, y)
        print('     > fitting the optimizer model...')
        xx = problem.sample(n=acquisition_samples)
        yy = acquisition.value(mu=surrogate.mean(xx), sigma=surrogate.std(xx))
        optimizer.fit(xx, yy)
        print('     > optimizing the acquisition function...')
        backend = cplex.CplexBackend()
        model = backend.new_model()
        network = optimizer.embed()
        x_var = [model.continuous_var(lb=lb, ub=ub) for lb, ub in zip(problem.lower_bounds, problem.upper_bounds)]
        y_var = model.continuous_var(lb=-float('inf'), ub=float('inf'))
        encode(backend, network, model, x_var, y_var, name='encoding')
        model.minimize(y_var)
        backend.solve(model, time_limit)
        print('     > new point found:', end=' ')
        opt_x = np.array([x.solution_value for x in x_var])
        opt_y = problem.evaluate(opt_x)
        x = np.concatenate((x, [opt_x]))
        y = np.concatenate((y, [opt_y]))
        print(f'({opt_x}, {opt_y})\n')
        draw(xf=x, yf=y, xa=xx, ya=yy, it=i + 1)
    best_index = np.argmin(y)
    print(f'OPTIMAL VALUE FOUND AT:({x[best_index]}), ({y[best_index]})')
