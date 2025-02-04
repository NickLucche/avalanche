"""
The :py:mod:`training` module provides a generic continual learning training
class (:py:class:`BaseStrategy`) and implementations of the most common
CL strategies. These are provided either as standalone strategies in
:py:mod:`training.strategies` or as plugins (:py:mod:`training.plugins`) that
can be easily combined with your own strategy.
"""
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.evaluation.metrics.reward import moving_window_stat
from avalanche.logging import InteractiveLogger
from avalanche.logging.interactive_logging import TqdmWriteInteractiveLogger
from .plugins import EvaluationPlugin


default_logger = EvaluationPlugin(
    accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
    loggers=[InteractiveLogger()])

default_rl_logger = EvaluationPlugin(
                                     moving_window_stat('reward', window_size=10, stats=['mean', 'max', 'std']),
                                     moving_window_stat('reward', window_size=4, stats=['mean', 'std'], mode='eval'),
                                     moving_window_stat('ep_length', window_size=10, stats=['mean', 'max', 'std']),
                                     moving_window_stat('ep_length', window_size=4, stats=['mean', 'std'], mode='eval'),
                                     loggers=[TqdmWriteInteractiveLogger(log_every=10)])
