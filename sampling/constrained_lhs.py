import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from docplex.mp.model import Model
from matplotlib.ticker import FormatStrFormatter


def plot(samples, limits, infeasible):
    bins = len(samples)
    sns.set_style('whitegrid')
    sns.set_context('notebook')
    ax = sns.histplot(x=samples[:, 0], y=samples[:, 1], bins=bins, binrange=limits, color='tab:orange')
    sns.histplot(x=infeasible[:, 0], y=infeasible[:, 1], bins=bins, binrange=limits, color='tab:blue')
    ax.scatter(x=samples[:, 0], y=samples[:, 1], s=10, color='black')
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_xticks(np.linspace(*limits[0], num=bins + 1, endpoint=True))
    ax.set_yticks(np.linspace(*limits[1], num=bins + 1, endpoint=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.show()


if __name__ == '__main__':
    # create centered grid of possible samples
    num = 10
    bounds = np.array([[-5, 10], [2, 6]])
    grid = [np.linspace(lb + (ub - lb) / (2 * num), ub - (ub - lb) / (2 * num), num=num) for lb, ub in bounds]
    grid = np.array(np.meshgrid(*grid))

    # precompute infeasible samples based on constraints
    constraints = [lambda v: v[0] * v[1] > -10, lambda v: v[0]**2 + v[1]**2 < 30]
    feasible = np.all([constraint(grid) for constraint in constraints], axis=0)

    # create model variables and eventually replace them with explicit zeros when samples are infeasible
    model = Model()
    variables = np.array(list(model.binary_var_matrix(num, num).values())).reshape((num, num))
    variables[~feasible] = 0.0

    # add latin hypercube sampling constraints (i.e., different rows and different columns)
    # these limitations are added as a cost function because the domain constraints may prevent lhs feasibility
    model.add_constraint(model.sum(variables) == num)
    costs = []
    for i in range(num):
        costs.append(model.sum(variables[i, :]) ** 2)
        costs.append(model.sum(variables[:, i]) ** 2)
    model.minimize(model.sum(costs))
    solution = model.solve()

    # retrieve the chosen indices and use them to mask the input values
    z = np.array([[v if isinstance(v, float) else v.solution_value for v in row] for row in variables])
    z = z.reshape((num, num)).astype(bool)
    x = grid.transpose((1, 2, 0))[z]
    plot(samples=x, limits=bounds, infeasible=grid.transpose((1, 2, 0))[~feasible])
