import numpy as np
import pandas as pd
import plotnine
import torch
from imageio import imwrite
from PIL import Image

from .analyze import Analysis, _register_analysis
from .gene_maps import generate_gene_maps
from ..data import Data, Dataset
from ..data.utility.misc import make_dataloader
from ..session import Session, require
from ..utility.file import chdir
from ..utility.mask import margin
from ..utility.tensor import sparseonehot, to_device
from ..utility.visualization import _normalize


def _save_samples(
    label: torch.Tensor,
    samples: torch.Tensor,
    ground_truth: torch.Tensor,
    save_raw_samples: bool = False,
) -> None:
    genes = require("genes")

    def _convolve(x: torch.Tensor) -> torch.Tensor:
        label_onehot = sparseonehot(label.flatten().long()).t().float()
        return label_onehot.mm(x.reshape(-1, x.shape[-1]).float())

    def _deconvolve(x: torch.Tensor) -> torch.Tensor:
        label_onehot = sparseonehot(label.flatten().long()).float()
        label_sizes = torch.sparse.sum(label_onehot, dim=0).to_dense()
        return (
            label_onehot.mm(
                x.reshape(-1, x.shape[-1]).float() / label_sizes.unsqueeze(1)
            )
        ).reshape(*label.shape, x.shape[-1])

    convolved_samples = torch.stack(
        [_convolve(x) for x in samples.permute(1, 2, 3, 0)]
    )
    convolved_ground_truth = _convolve(ground_truth)

    torch.save(label, "labels.pkl")

    def _make_df(x: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            data=x,
            columns=genes,
            index=pd.Series(np.arange(x.shape[0]), name="n"),
        )

    summaries = {
        "mean": convolved_samples.mean(0).cpu().numpy(),
        "stdv": convolved_samples.std(0).cpu().numpy(),
        "log1pstdv": convolved_samples.std(0).log1p().cpu().numpy(),
        "q05": np.quantile(convolved_samples.cpu().numpy(), 0.05, axis=0),
        "q50": np.quantile(convolved_samples.cpu().numpy(), 0.50, axis=0),
        "q95": np.quantile(convolved_samples.cpu().numpy(), 0.95, axis=0),
        "gt": convolved_ground_truth.cpu().numpy(),
    }

    dfs, names = zip(
        *[(_make_df(x).loc[1:], name) for name, x in summaries.items()]
    )
    df = pd.concat(dfs, axis=1, keys=names)
    df.columns = df.columns.map("_".join)
    df.to_parquet("prediction.parquet", compression="none")

    for summary_name, x in {
        "error": np.abs(summaries["mean"] - summaries["gt"]),
        "log1perror": np.log1p(np.abs(summaries["mean"] - summaries["gt"])),
        **summaries,
    }.items():
        summary_image = (
            _deconvolve(to_device(torch.as_tensor(x))).cpu().numpy()
        )
        summary_image = _normalize(summary_image)
        summary_image = np.concatenate(
            [summary_image, (1.0 - summary_image.max(-1, keepdims=True)),], -1,
        )
        summary_image = (255 * summary_image).astype(np.uint8)
        summary_image = Image.fromarray(summary_image, mode="CMYK")
        summary_image = summary_image.convert("RGB")
        imwrite(
            f"{summary_name}.png", summary_image,
        )

    long_df = pd.wide_to_long(
        df.reset_index(),
        ["mean", "stdv", "q05", "q50", "q95", "gt"],
        i="n",
        j="mol",
        sep="_",
        suffix=r"mol\d+",
    )
    long_df = long_df.reset_index()
    uncertainty_plot = (
        plotnine.ggplot(long_df)
        + plotnine.aes(
            "np.log1p(stdv)",
            "mean - gt",
            ymin="q05 - gt",
            ymax="q95 - gt",
            color="mol",
        )
        + plotnine.geom_point()
        + plotnine.geom_errorbar(width=0)
    )
    uncertainty_plot.save("uncertainty.png",)

    if save_raw_samples:
        (
            pd.concat(
                [
                    pd.DataFrame(
                        x.cpu().numpy(),
                        columns=genes,
                        index=pd.Series(np.arange(len(x)), name="n"),
                    )
                    .assign(sample=i)
                    .reset_index()
                    for i, x in enumerate(convolved_samples, 1)
                ],
                ignore_index=True,
            ).to_parquet("samples.parquet", compression="none", index=False)
        )


def compute_synthetic(
    num_samples: int = 20,
    predict_mean: bool = True,
    save_raw_samples: bool = False,
    test_grid_size: float = 32,
    trim_margins: bool = False,
) -> None:
    # pylint: disable=too-many-locals
    dataset = require("dataloader").dataset
    genes = require("genes")

    for slide_name, slide in dataset.data.slides.items():
        slideloader = make_dataloader(
            Dataset(
                Data(
                    slides={slide_name: slide},
                    design=dataset.data.design.loc[[slide_name]],
                )
            ),
            batch_size=1,
            shuffle=False,
        )
        with Session(dataloader=slideloader, eval=True):
            sampled_genes, samples = zip(
                *[
                    (gene, samples)
                    for _, gene, samples in generate_gene_maps(
                        num_samples=num_samples,
                        genes_per_batch=len(genes),
                        predict_mean=predict_mean,
                        scale=1.0,
                    )
                ]
            )
        assert list(sampled_genes) == genes

        samples = to_device(torch.stack(samples))
        label = to_device(
            torch.as_tensor(dataset.data.slides[slide_name].data.label[()])
        )
        ground_truth = to_device(
            torch.as_tensor(
                dataset.data.slides[slide_name].data.extra("ground_truth")
            )
        )

        if trim_margins:
            row_mask, col_mask = to_device(
                torch.as_tensor(
                    margin(
                        label.cpu().numpy(), margin_color=np.array(0), tol=0.0
                    )
                )
            )
            label = label[~row_mask][:, ~col_mask]
            ground_truth = ground_truth[~row_mask][:, ~col_mask]
            samples = samples[:, :, ~row_mask][:, :, :, ~col_mask]

        with chdir(f"{slide_name}/observed"):
            _save_samples(
                label, samples, ground_truth, save_raw_samples=save_raw_samples
            )

        with chdir(f"{slide_name}/pixel"):
            pixel_map = (
                torch.arange(1, 1 + np.prod(label.shape))
                .to(label)
                .reshape_as(label)
            )
            _save_samples(
                pixel_map,
                samples,
                ground_truth,
                save_raw_samples=save_raw_samples,
            )

        with chdir(slide_name):
            imwrite("image.png", slide.data.image)

            label_map = pd.DataFrame(
                {
                    "observed": label[label != 0].flatten().cpu(),
                    "pixel": pixel_map[label != 0].flatten().cpu(),
                }
            )
            label_map.to_parquet("label_map.parquet", compression="none")

            count_sums = pd.DataFrame(
                samples.reshape(samples.shape[0], samples.shape[1], -1)
                .sum(-1)
                .t()
                .cpu()
                .numpy(),
                columns=genes,
            ).melt(var_name="mol", value_name="reads")
            count_sums.to_parquet(
                "count_sums.parquet", compression="none",
            )
            count_sums_gt = pd.DataFrame(
                ground_truth.reshape(-1, ground_truth.shape[-1])
                .sum(0)
                .unsqueeze(0)
                .cpu()
                .numpy(),
                columns=genes,
            ).melt(var_name="mol", value_name="reads")
            count_sums_gt.to_parquet(
                "count_sums_gt.parquet", compression="none",
            )
            count_sums_plot = (
                plotnine.ggplot(count_sums)
                + plotnine.aes("reads", fill="mol")
                + plotnine.geom_histogram(bins=100)
                + plotnine.geom_vline(
                    plotnine.aes(xintercept="reads", color="mol"),
                    data=count_sums_gt,
                    linetype="dashed",
                )
                + plotnine.coords.coord_cartesian(xlim=(0, None))
            )
            count_sums_plot.save("count_sums.png")


_register_analysis(
    name="synthetic",
    analysis=Analysis(
        description="Synthetic data tests", function=compute_synthetic,
    ),
)
