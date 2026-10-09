"""Stateless physical features derived only from permitted sensor readings."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from machine_failure.data import FEATURE_COLUMNS, select_features

PHYSICAL_COLUMNS = ("Temperature difference [K]", "Mechanical power [W]", "Torque wear [Nm min]")


class PhysicalFeatures(TransformerMixin, BaseEstimator):
    def fit(self, features, target=None):
        self.n_features_in_ = len(FEATURE_COLUMNS)
        self.feature_names_in_ = np.array(FEATURE_COLUMNS, dtype=object)
        return self

    def transform(self, features):
        result = select_features(features)
        result[PHYSICAL_COLUMNS[0]] = (
            result["Process temperature [K]"] - result["Air temperature [K]"]
        )
        result[PHYSICAL_COLUMNS[1]] = (
            result["Torque [Nm]"] * result["Rotational speed [rpm]"] * 2 * np.pi / 60
        )
        result[PHYSICAL_COLUMNS[2]] = result["Torque [Nm]"] * result["Tool wear [min]"]
        return result
