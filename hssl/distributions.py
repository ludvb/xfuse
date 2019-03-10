from abc import abstractmethod

import torch as t

from .logging import DEBUG, log


class Distribution:
    def __init__(self):
        self._names = [a for a, _ in self.parameters]
        for n in self._names:
            try:
                _ = self._get_r_transformer(n)
            except AttributeError:
                log(DEBUG, '%s parameter "%s" is defined on R',
                    type(self).__name__, n)
                self._set_r_transformer(n, lambda x: x)

    def _set_r_transformer(self, name, f):
        setattr(self, f'_set_{name}', f)
        return self

    def _get_r_transformer(self, name):
        return getattr(self, f'_set_{name}')

    def set_parameter(self, name, value, r_transform=False):
        if name not in self._names:
            raise ValueError(
                f'attempted to set non-existent parameter "{name}" '
                f'(available: {", ".join(self._names)})'
            )
        setattr(self, name, (
            self._get_r_transformer(name)(value)
            if r_transform
            else value
        ))
        return self

    def set(self, r_transform=False, **kwargs):
        for p, v in kwargs.items():
            self.set_parameter(p, v, r_transform)
        return self

    @property
    def parameters(self):
        pass

    @abstractmethod
    def log_prob(self, x):
        pass

    @abstractmethod
    def sample(self):
        pass

    @property
    def mean(self):
        pass

    def __dict__(self):
        return self.parameters


class Variable:
    def __init__(self, distribution: Distribution):
        self.distribution = distribution
        self.value = None

    def sample(self):
        self.value = self.distribution.sample()
        return self

    @property
    def log_prob(self):
        return self.distribution.log_prob(self.value)


class Deterministic(Distribution):
    def __init__(self, value=None):
        self.x = t.as_tensor(value or 0.)
        super().__init__()

    @property
    def parameters(self):
        return [('x', self.x)]

    def log_prob(self, x):
        return t.prod(x == self.x)

    def sample(self):
        return self.x

    @property
    def mean(self):
        return self.x


class NegativeBinomial(Distribution):
    def __init__(self, rate=None, logit=None):
        self.rate = rate
        self.logit = logit
        super().__init__()

    @property
    def parameters(self):
        return [
            ('rate', self.rate),
            ('logit', self.logit),
        ]

    def _set_rate(self, x):
        return t.exp(x) + 1e-10

    def log_prob(self, x):
        return (
            t.distributions.NegativeBinomial(self.rate, logits=self.logit)
            .log_prob(x)
            .sum()
        )

    def sample(self):
        return (
            t.distributions.NegativeBinomial(self.rate, logits=self.logit)
            .rsample()
        )

    @property
    def mean(self):
        return self.rate * t.exp(self.logit)


class Normal(Distribution):
    def __init__(self, loc=None, scale=None):
        if loc is None: loc = 0.
        if scale is None: scale = 1.
        self.loc = t.as_tensor(loc)
        self.scale = t.as_tensor(scale)
        super().__init__()

    @property
    def parameters(self):
        return [
            ('loc', self.loc),
            ('scale', self.scale),
        ]

    def _set_scale(self, x):
        return t.abs(x)

    def log_prob(self, x):
        return (
            t.distributions.Normal(self.loc, self.scale)
            .log_prob(x)
            .sum()
        )

    def sample(self):
        return (
            t.distributions.Normal(self.loc, self.scale)
            .rsample()
        )

    @property
    def mean(self):
        return self.loc
