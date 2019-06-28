from .stats_handler import StatsHandler


class Image(StatsHandler):
    def _select_msg(self, name, is_observed, **_):
        return is_observed and name[-5:] == 'image'

    def _handle(self, fn, value, **_):
        self.add_images('image/ground_truth', (1 + value) / 2)
        self.add_images('image/mean', (1 + fn.mean) / 2)
        self.add_images('image/sample', (1 + fn.sample()) / 2)
