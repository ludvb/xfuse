from abc import abstractmethod

from .stats_handler import StatsHandler
from ...logging import WARNING, log
from ...model.experiment.st import ST
from ...utility.visualization import reduce_last_dimension
from ...session import get_model


__all__ = [
    'FactorActivationHistogram',
    'FactorActivationMaps',
    'FactorActivationMean',
    'FactorActivationSummary',
]


class FactorActivation(StatsHandler):
    def _select_msg(self, name, **_):
        return name[-3:] == 'rim'

    @abstractmethod
    def _handle_factor_activation(self, name, activation):
        pass

    def _handle(self, fn, **_):
        try:
            experiment = get_model()._get_experiment('ST')
        except KeyError:
            log(WARNING, 'can\'t track factor activation:'
                         'model does not have an ST experiment')
            return None
        for name, factor in zip(
                experiment.factors, fn.mean.permute(1, 0, 2, 3)):
            self._handle_factor_activation(name, factor)


class FactorActivationHistogram(FactorActivation):
    def _handle_factor_activation(self, name, activation):
        self.add_histogram(f'activation/factor{name}', activation.flatten())


class FactorActivationMaps(FactorActivation):
    def _handle_factor_activation(self, name, activation):
        self.add_images(
            f'activation/maps/factor{name}',
            activation.unsqueeze(1),
            dataformats='NCHW',
        )


class FactorActivationMean(FactorActivation):
    def _handle_factor_activation(self, name, activation):
        self.add_scalar(f'activation/mean/factor{name}', activation.mean())


class FactorActivationSummary(StatsHandler):
    def _select_msg(self, name, value, **_):
        return name[-3:] == 'rim' and value.shape[1] >= 3

    def _handle(self, fn, **_):
        self.add_images(
            'activation/summary',
            reduce_last_dimension(fn.mean.permute(0, 2, 3, 1)),
            dataformats='NHWC',
        )
