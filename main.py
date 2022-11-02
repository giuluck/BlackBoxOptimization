import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from eml.backend import cplex_backend as cplex

from acquisitions import *
from models import *
from problems import *

problem = AckleyProblem((-5, 10))
# surrogate = NeuralNetwork(units=[8, 8, 8], epochs=5000, std=Model.gp_std)
surrogate = GaussianProcess(normalize_y=True)
# optimizer = NeuralNetwork(units=[8, 8, 8], epochs=5000)
optimizer = RegressionTree()
acquisition = ExpectedImprovement()
use_weights = False
iterations = 1
initial_samples = 4
acquisition_samples = 10
time_limit = 30
seed = 0


def draw(xf, yf, xa, ya, it):
    sns.set_context('notebook')
    sns.set_style('whitegrid')
    # build linear space for drawing
    space = np.linspace(problem.lower_bounds, problem.upper_bounds, endpoint=True, num=1000)
    mu = surrogate.mean(space)
    sigma = surrogate.std(space)
    best = np.max(yf[:-1])
    acq = acquisition.value(mu, sigma, best)
    plt.figure(figsize=(16, 9), tight_layout=True)
    # plot true function
    ax = plt.subplot(221)
    plt.plot(space, problem.evaluate(space), zorder=1)
    plt.scatter(xf[-1:], yf[-1:], color='red', zorder=2, label='optima')
    plt.scatter(xf[:-1], yf[:-1], color='black', zorder=3, label='points')
    plt.title('True Function')
    plt.legend()
    # plot surrogate model
    plt.subplot(222, sharex=ax, sharey=ax)
    plt.plot(space, mu, zorder=1)
    plt.fill_between(space[:, 0], mu - sigma, mu + sigma, alpha=0.3, zorder=1)
    plt.scatter(xf[-1:], yf[-1:], color='red', zorder=2, label='optima')
    plt.scatter(xf[:-1], yf[:-1], color='black', zorder=3, label='points')
    plt.vlines(xf[-1:], yf[-1:], surrogate.mean(xf[-1:]), linestyles='--', color='red', zorder=2)
    plt.vlines(xf[:-1], yf[:-1], surrogate.mean(xf[:-1]), linestyles='--', color='black', zorder=3)
    plt.title('Surrogate Function')
    plt.legend()
    # plot acquisition function
    mxf = surrogate.mean(xf[-1:])
    sxf = surrogate.std(xf[-1:])
    ayf = acquisition.value(mxf, sxf, best)
    ax = plt.subplot(223)
    plt.plot(space, acq, zorder=1)
    plt.scatter(xf[-1:], ayf, color='red', zorder=2, label='optima')
    plt.scatter(xa, ya, color='black', s=5, zorder=3, label='points')
    plt.title('Acquisition Function')
    plt.legend()
    # plot surrogate acquisition
    plt.subplot(224, sharex=ax, sharey=ax)
    plt.plot(space, optimizer.mean(space), zorder=1)
    plt.scatter(xf[-1:], ayf, color='red', zorder=2, label='optima')
    plt.scatter(xa, ya, color='black', s=5, zorder=3, label='points')
    plt.vlines(xf[-1:], ayf, optimizer.mean(xf[-1:]), linestyles='--', color='red', zorder=2)
    plt.vlines(xa, ya, optimizer.mean(xa), alpha=0.2, linestyles='--', color='black', zorder=3)
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
        yy = acquisition.value(mu=surrogate.mean(xx), sigma=surrogate.std(xx), best=np.max(y))
        optimizer.fit(xx, yy, sample_weight=yy if use_weights else None)
        print('     > optimizing the acquisition function...')
        backend = cplex.CplexBackend()
        model = backend.new_model()
        x_var = [model.continuous_var(lb=lb, ub=ub) for lb, ub in zip(problem.lower_bounds, problem.upper_bounds)]
        y_var = model.continuous_var(lb=-float('inf'), ub=float('inf'))
        optimizer.encode(backend, model, x_var, y_var)
        model.maximize(y_var)
        backend.solve(model, time_limit)
        print('     > new point found:', end=' ')
        opt_x = np.array([x.solution_value for x in x_var])
        opt_y = problem.evaluate(opt_x)
        x = np.concatenate((x, [opt_x]))
        y = np.concatenate((y, [opt_y]))
        print(f'({opt_x}, {opt_y})\n')
        draw(xf=x, yf=y, xa=xx, ya=yy, it=i + 1)
    best_index = np.argmax(y)
    print(f'OPTIMAL VALUE FOUND AT:({x[best_index]}), ({y[best_index]})')
