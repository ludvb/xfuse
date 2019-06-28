from abc import abstractmethod

from typing import List

from .stats_handler import StatsHandler
from ...utility import reduce_last_dimension


__all__ = [
    'FactorActivationMean',
    'FactorActivationMaps',
    'FactorActivationSummary',
    'FactorActivationHistogram',
]


class FactorActivation(StatsHandler):
    def __init__(self, factor_names: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._factor_names = factor_names

    def _select_msg(self, name, **_):
        return name[-3:] == 'rim'

    @abstractmethod
    def _handle_factor_activation(self, name, activation):
        pass

    def _handle(self, fn, **_):
        for name, factor in zip(
                self._factor_names, fn.mean.permute(1, 0, 2, 3)):
            self._handle_factor_activation(name, factor)


class FactorActivationMean(FactorActivation):
    def _handle_factor_activation(self, name, activation):
        self.add_scalar(f'activation/mean/factor{name}', activation.mean())


class FactorActivationMaps(FactorActivation):
    def _handle_factor_activation(self, name, activation):
        self.add_images(
            f'activation/maps/factor{name}',
            activation.unsqueeze(1),
            dataformats='NCHW',
        )


class FactorActivationSummary(StatsHandler):
    def _select_msg(self, name, value, **_):
        return name[-3:] == 'rim' and value.shape[1] >= 3

    def _handle(self, fn, **_):
        self.add_images(
            'activation/summary',
            reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
            dataformats='NHWC',
        )


class FactorActivationHistogram(FactorActivation):
    def _handle_factor_activation(self, name, activation):
        self.add_histogram(f'activation/factor{name}', activation.flatten())
