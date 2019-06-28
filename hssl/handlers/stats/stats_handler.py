from abc import abstractmethod

from pyro.poutine.messenger import Messenger

from torch.utils.tensorboard.writer import SummaryWriter


class StatsHandler(Messenger):
    def __init__(self, writer: SummaryWriter, global_step: int):
        from functools import wraps
        import inspect

        self.__writer = writer
        self._global_step = global_step

        def _add_writer_method(name, method):
            setattr(
                self,
                name,
                wraps(method)(
                    lambda *args, **kwargs: method(
                        *args, global_step=self._global_step, **kwargs)
                    if "global_step" in inspect.signature(method).parameters
                    else
                    lambda *args, **kwargs: method(*args, **kwargs)
                ),
            )

        for name, method in (
                (name, attr)
                for name, attr in (
                        (name, getattr(self.__writer, name))
                        for name in dir(self.__writer)
                )
                if name[0] != '_' if callable(attr)
        ):
            _add_writer_method(name, method)

    @abstractmethod
    def _handle(self, **msg) -> None:
        pass

    @abstractmethod
    def _select_msg(self, **msg) -> bool:
        pass

    def _postprocess_message(self, msg):
        if msg['type'] == 'sample' and self._select_msg(**msg):
            self._handle(**msg)
