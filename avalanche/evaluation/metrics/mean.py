################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 19-01-2021                                                             #
# Author(s): Lorenzo Pellegrini                                                #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from typing import SupportsFloat

from avalanche.evaluation import Metric
from typing import List
import numpy as np

class Mean(Metric[float]):
    """
    The standalone mean metric.

    This utility metric is a general purpose metric that can be used to keep
    track of the mean of a sequence of values.
    """
    def __init__(self):
        """
        Creates an instance of the mean metric.

        This metric in its initial state will return a mean value of 0.
        The metric can be updated by using the `update` method while the mean
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.summed: float = 0.0
        self.weight: float = 0.0

    def update(self, value: SupportsFloat, weight: SupportsFloat = 1.0) -> None:
        """
        Update the running mean given the value.

        The value can be weighted with a custom value, defined by the `weight`
        parameter.

        :param value: The value to be used to update the mean.
        :param weight: The weight of the value. Defaults to 1.
        :return: None.
        """
        value = float(value)
        weight = float(weight)
        self.summed += value * weight
        self.weight += weight

    def result(self) -> float:
        """
        Retrieves the mean.

        Calling this method will not change the internal state of the metric.

        :return: The mean, as a float.
        """
        if self.weight == 0.0:
            return 0.0
        return self.summed / self.weight

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.summed = 0.0
        self.weight = 0.0

class WindowedMovingAverage(Metric[float]):

    def __init__(self, window_size: int):
        assert window_size > 0, "Window size cannot be negative"
        super().__init__()
        self.window:List[float] = []
        # useful to compute additional stats on window
        self.window_size = window_size

    def update(self, value: SupportsFloat):
        value = float(value)
        self.window.append(value)
        if len(self.window) > self.window_size:
            self.window = self.window[1:]

    def result(self)->float:
        if not len(self.window):
            return np.float('-inf')
        return sum(self.window)/len(self.window)

    def reset(self):
        self.window = []


class Sum(Metric[float]):
    """
    The standalone sum metric.

    This utility metric is a general purpose metric that can be used to keep
    track of the sum of a sequence of values.

    Beware that this metric only supports summing numbers and the result is
    always a float value, even when `update` is called by passing `int`s only.
    """

    def __init__(self):
        """
        Creates an instance of the sum metric.

        This metric in its initial state will return a sum value of 0.
        The metric can be updated by using the `update` method while the sum
        can be retrieved using the `result` method.
        """
        super().__init__()
        self.summed: float = 0.0

    def update(self, value: SupportsFloat) -> None:
        """
        Update the running sum given the value.

        :param value: The value to be used to update the sum.
        :return: None.
        """
        self.summed += float(value)

    def result(self) -> float:
        """
        Retrieves the sum.

        Calling this method will not change the internal state of the metric.

        :return: The sum, as a float.
        """
        return self.summed

    def reset(self) -> None:
        """
        Resets the metric.

        :return: None.
        """
        self.summed = 0.0


__all__ = ['Mean', 'Sum']
