import warnings

import pyro
import torch
from imageio import imwrite

from ..data import Data, Dataset
from ..data.slide import AnnotatedImage, FullSlideIterator, Slide
from ..data.utility.misc import make_dataloader
from ..messengers.stats.stats_handler import log_image
from ..session import Session, require
from ..utility.file import chdir
from ..utility.tensor import to_device, sparseonehot
from ..utility.visualization import _normalize
from .analyze import Analysis, _register_analysis
from .gene_maps import _compute_grid_annotation


class ResampleFromPrior(pyro.poutine.messenger.Messenger):
    def _pyro_post_sample(self, msg):
        try:
            msg["value"] = (
                msg["fn"]
                .to_event(-len(msg["fn"].shape()) + 1)
                .expand(msg["value"].shape)
                .sample()
            )
        except ValueError:
            warnings.warn(f'Failed to resample "{msg["name"]}"')
        return msg


def _run_dream(data):
    genes = require("genes")
    model = require("model")

    # pylint: disable=fixme
    # FIXME: Compatibility hack.
    #        Let's get back to this when reworking the model code.
    st_data = {
        "slide": data["slide"],
        "covariates": [
            {
                covariate: condition
                for covariate, condition in covariates.items()
            }
            for covariates in data["covariates"]
        ],
        "data": [
            to_device(
                torch.zeros(
                    int(label.max().item()), len(genes), dtype=torch.float32,
                )
            )
            for label in data["label"]
        ],
        "label": data["label"],
        "image": data["image"],
    }

    with Session(eval=True):
        with pyro.poutine.trace() as guide_trace:
            model.guide({"ST": st_data})

    with Session(eval=True):
        with ResampleFromPrior(), pyro.poutine.replay(trace=guide_trace.trace):
            with pyro.poutine.trace() as model_trace:
                model({"ST": st_data})

    return model_trace.trace


def _sleep() -> None:
    """Runs sleep"""

    dataloader = require("dataloader")
    model = require("model")

    for slide_name, slide in dataloader.dataset.data.slides.items():
        annotation, label_names = _compute_grid_annotation(
            slide.data.label.shape
        )
        slideloader = make_dataloader(
            Dataset(
                Data(
                    slides={
                        slide_name: Slide(
                            data=AnnotatedImage(
                                torch.as_tensor(slide.data.image),
                                annotation=annotation,
                                name="coordinates",
                                label_names=label_names,
                            ),
                            iterator=FullSlideIterator,
                        )
                    },
                    design=dataloader.dataset.data.design.loc[[slide_name]],
                )
            ),
            batch_size=1,
            shuffle=False,
        )
        try:
            data = next(iter(slideloader))
        except StopIteration as exc:
            raise RuntimeError() from exc

        data = to_device(data["AnnotatedImage"])

        trace = _run_dream(data)

        with chdir(slide_name):
            dream_image = (
                trace.nodes["ST/image"]["fn"]
                .sample()[0]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            dream_image = _normalize(dream_image)
            imwrite("image.png", dream_image)
            log_image("dream/image", dream_image)

            dream_expression = to_device(
                torch.zeros((*data["label"].shape, 3)).float()
            )
            dream_expression[data["label"] != 0] = (
                sparseonehot(
                    (data["label"][data["label"] != 0] - 1).flatten().long()
                )
                .float()
                .mm(trace.nodes["ST/xsg-0"]["fn"].mean)
            )
            dream_expression = dream_expression.squeeze().cpu().numpy()
            dream_expression = _normalize(dream_expression)
            log_image("dream/expression", dream_expression)
            imwrite("expression.png", dream_expression)


_register_analysis(
    name="dream", analysis=Analysis(description="?", function=_sleep,),
)
