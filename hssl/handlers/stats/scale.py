from .stats_handler import StatsHandler


class Scale(StatsHandler):
    def _select_msg(self, name, **_):
        return name[-5:] == "scale"

    def _handle(self, fn, **_):
        self.add_images(
            "scale", fn.mean.permute(0, 2, 3, 1), dataformats="NHWC"
        )
