from typing import Any, Optional, Callable

import numpy as np
from eml.tree.reader import sklearn_reader
from eml.tree.embed import encode_backward_implications
from sklearn.tree import DecisionTreeRegressor

from models.model import EncodeModel, VarianceModel


class RegressionTree(VarianceModel, EncodeModel):
    def __init__(self, std: Callable = EncodeModel.gp_std, criterion='squared_error', **tree_kwargs):
        """A Decision Tree Regression model."""
        super(RegressionTree, self).__init__(std=std)

        self.model = DecisionTreeRegressor(criterion=criterion, **tree_kwargs)
        "The tree model."

    def fit(self, x: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model.fit(x, y, sample_weight=sample_weight)

    def mean(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(x).flatten()

    def encode(self, backend, model, x_var, y_var, name='encoding') -> Any:
        tree = sklearn_reader.read_sklearn_tree(self.model)
        for i, v in enumerate(x_var):
            tree.update_lb(i, v.lb)
            tree.update_ub(i, v.ub)
        return encode_backward_implications(backend, tree, model, x_var, y_var, name=name)
