from abc import abstractmethod

import torch as t

from .logging import DEBUG, log


class Distribution:
    def __init__(self):
        def _create_getter(name):
            @property
            def _get(self):
                value = t.as_tensor(self._get_raw_value(name))
                transformer = self._get_r_transformer(name)
                return transformer(value)
            return _get

        for n in self.parameters:
            try:
                transformer = self._get_default_r_transformer(n)
            except AttributeError:
                log(DEBUG, '%s parameter "%s" is defined on R',
                    type(self).__name__, n)
                transformer = lambda x: x
                self._set_default_r_transformer(n, transformer)
            self._set_r_transformer(n, transformer)
            setattr(self.__class__, n, _create_getter(n))

    def _set_r_transformer(self, name, f):
        setattr(self, f'_rtrans_{name}', f)
        return self

    def _get_r_transformer(self, name):
        return getattr(self, f'_rtrans_{name}')

    def _set_default_r_transformer(self, name, value):
        setattr(self, f'_set_{name}', value)
        return self

    def _get_default_r_transformer(self, name):
        return getattr(self, f'_set_{name}')

    def _set_raw_value(self, name, value):
        setattr(self, f'_{name}', value)
        return self

    def _get_raw_value(self, name):
        return getattr(self, f'_{name}')

    def set_parameter(self, name, value, r_transform=False):
        if name not in self.parameters:
            raise ValueError(
                f'attempted to set non-existent parameter "{name}" '
                f'(available: {", ".join(self._names)})'
            )
        self._set_raw_value(name, value)
        self._set_r_transformer(
            name,
            (
                self._get_default_r_transformer(name)
                if r_transform
                else lambda x: x
            ),
        )
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
        super().__init__()
        if value is None: value = 0.
        self.set_parameter('x', value)

    @property
    def parameters(self):
        return ['x']

    def log_prob(self, x):
        return t.prod(x == self.x).float()

    def sample(self):
        return self.x

    @property
    def mean(self):
        return self.x


class NegativeBinomial(Distribution):
    def __init__(self, rate=None, logit=None):
        super().__init__()
        self.set_parameter('rate', rate)
        self.set_parameter('logit', logit)

    @property
    def parameters(self):
        return ['rate', 'logit']

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
        super().__init__()
        if loc is None: loc = 0.
        if scale is None: scale = 1.
        self.set_parameter('loc', loc)
        self.set_parameter('scale', scale)

    @property
    def parameters(self):
        return ['loc', 'scale']

    def _set_scale(self, x):
        return (
            t.nn.functional.softplus(x)
            .clamp_min(1e-10)
        )

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
