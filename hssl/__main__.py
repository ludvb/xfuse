# pylint: disable=missing-docstring, invalid-name, too-many-instance-attributes

from functools import partial

from datetime import datetime as dt

from functools import wraps

from inspect import getargs

import itertools as it

import os

import sys

import click

import numpy as np

import pandas as pd

from pyvips import Image

import torch as t

from . import __version__
from .analyze import (
    Sample,
    analyze as default_analysis,
    analyze_gene_profiles,
    analyze_genes,
    dge as dge_analysis,
    impute_counts,
)
from .dataset import Dataset, RandomSlide, spot_size
from .logging import (
    DEBUG,
    ERROR,
    INFO,
    WARNING,
    LoggedExecution,
    log,
    set_level,
)
from .network import (
    XFuse,
    STEncoder,
    STDecoder,
    HEEncoder,
    HEDecoder,
)
from .optimizer import create_optimizer
from .train import train as _train
from .utility import (
    compose,
    design_matrix_from,
    read_data,
    set_rng_seed,
)
from .utility.state import (
    State,
    load_state,
    save_state,
    to_device,
)


DEVICE = t.device('cuda' if t.cuda.is_available() else 'cpu')


def _logged_command(get_output_dir):
    def _decorator(f):
        @wraps(f)
        def _wrapper(*args, **kwargs):
            output_dir = get_output_dir(*args, **kwargs)

            if os.path.exists(output_dir):
                log(ERROR, 'output directory %s already exists', output_dir)
                sys.exit(1)

            os.makedirs(output_dir)

            with LoggedExecution(os.path.join(output_dir, 'log')):
                log(INFO, 'this is %s %s', __package__, __version__)
                log(DEBUG, 'invoked by %s', ' '.join(sys.argv))
                log(INFO, 'device: %s', str(DEVICE))

                f(*args, **kwargs)
        return _wrapper
    return _decorator


@click.group()
@click.option('-v', '--verbose', is_flag=True)
@click.version_option()
def cli(verbose):
    if verbose:
        set_level(DEBUG)
    else:
        set_level(INFO)


@click.command()
@click.argument('design-file', type=click.File('rb'))
@click.option('--factors', type=int, default=50)
@click.option('--patch-size', type=int, default=512)
@click.option('--lr', type=float, default=1e-3)
@click.option('--batch-size', type=int, default=8)
@click.option('--workers', type=int)
@click.option('--seed', type=int)
@click.option(
    '--restore',
    'state_file',
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.option(
    '-o', '--output',
    type=click.Path(resolve_path=True),
    default=f'{__package__}-{dt.now().isoformat()}',
)
@click.option('--checkpoint', 'chkpt_interval', type=int)
@click.option('--image', 'image_interval', type=int, default=1000)
@click.option('--epochs', type=int)
@_logged_command(lambda *_, **args: args['output'])
def train(
        design_file,
        factors,
        patch_size,
        lr,
        output,
        state_file,
        workers,
        seed,
        **kwargs,
):
    if seed is not None:
        set_rng_seed(seed)

        if workers is None:
            log(WARNING,
                'setting workers to 0 to avoid race conditions '
                '(set --workers explicitly to override)')
            workers = 0

    design = pd.read_csv(design_file)
    design_dir = os.path.dirname(design_file.name)

    def _path(p):
        return (
            p
            if os.path.isabs(p) else
            os.path.join(design_dir, p)
        )

    count_data = read_data(map(_path, design.data))

    design_matrix = design_matrix_from(design[[
        x for x in design.columns
        if x not in [
                'name',
                'image',
                'labels',
                'validation',
                'data',
        ]
    ]])

    dataset = Dataset(
        [
            RandomSlide(
                data=counts,
                image=Image.new_from_file(_path(image)),
                label=Image.new_from_file(_path(labels)),
                patch_size=patch_size,
            )
            for image, labels, counts in zip(
                design.image,
                design.labels,
                (count_data.loc[x] for x in count_data.index.levels[0]),
            )
        ],
        design_matrix,
    )
    try:
        dataset_validation = Dataset(
            [
                RandomSlide(
                    data=counts,
                    image=Image.new_from_file(_path(image)),
                    label=Image.new_from_file(_path(labels)),
                    patch_size=patch_size,
                )
                for image, labels, counts in zip(
                    design.image,
                    design.validation,
                    (count_data.loc[x] for x in count_data.index.levels[0]),
                )
            ],
            design_matrix,
        )
    except AttributeError:
        dataset_validation = None

    state: State
    if state_file is not None:
        state = load_state(state_file)
    else:
        st_encoder = partial(
            STEncoder,
            genes=count_data.columns,
        )
        st_decoder = partial(
            STDecoder,
            genes=count_data.columns,
            covariates=[
                (k, set(v for k, v in items))
                for k, items in it.groupby(
                        design_matrix.index.to_flat_index().to_list(),
                        lambda x: x[0],
                )
            ],
            gene_baseline=count_data.mean(0).values,
        )

        he_encoder = HEEncoder
        he_decoder = HEDecoder

        model = XFuse(
            [st_encoder, he_encoder],
            [st_decoder, he_decoder],
            dataset_size=len(dataset),
        )

        optimizer = create_optimizer(model, learning_rate=lr)
        state = State(model, optimizer, 0)

    state = _train(
        state=state,
        output_prefix=output,
        dataset=dataset,
        dataset_validation=dataset_validation,
        workers=workers,
        **kwargs,
    )

    save_state(state, os.path.join(output, 'final-state.pkl'))


cli.add_command(train)


@click.group(chain=True)
@click.option(
    '--state-file',
    '--state',
    '--restore',
    type=click.File('rb'),
    required=True,
)
@click.option(
    '--design-file',
    '--design',
    type=click.File('rb'),
)
@click.option(
    '-o', '--output',
    type=click.Path(resolve_path=True),
    default=f'{__package__}-{dt.now().isoformat()}',
)
def analyze(**_):
    pass


@analyze.resultcallback()
@_logged_command(lambda *_, **args: args['output'])
def _run_analysis(analyses, design_file, state_file, output):
    state = load_state(state_file.name)
    to_device(state, DEVICE)

    t.no_grad()
    state.histonet.eval()
    state.std.eval()

    design = pd.read_csv(design_file)
    design_dir = os.path.dirname(design_file.name)

    def _path(p):
        return (
            p
            if os.path.isabs(p) else
            os.path.join(design_dir, p)
        )

    data = read_data(map(_path, design.data), genes=state.std.genes)
    design_matrix = design_matrix_from(design, state.std._covariates)
    samples = [
        Sample(
            name=name,
            image=image,
            label=label,
            data=data,
            effects=effects,
        )
        for name, image, label, data, effects in it.zip_longest(
                (
                    design.name
                    if 'name' in design.columns else
                    [f'sample_{i + 1}' for i in range(design.shape[0])]
                ),
                map(compose(Image.new_from_file, _path), design.image),
                (
                    map(
                        compose(Image.new_from_file, _path),
                        design.labels,
                    )
                    if 'labels' in design.columns else
                    []
                ),
                [data.xs(a, level=0) for a in data.index.levels[0]],
                design_matrix.values.transpose(),
        )
    ]

    for name, analysis in analyses:
        log(INFO, 'performing analysis: %s', name)
        if getargs(analysis.__code__).args == ['state', 'samples', 'output']:
            analysis(state=state, samples=samples, output=output)
        elif getargs(analysis.__code__).args == ['state', 'sample', 'output']:
            for sample in samples:
                log(INFO, 'processing %s', sample.name)
                output_prefix = os.path.join(output, sample.name)
                os.makedirs(output_prefix, exist_ok=True)
                analysis(
                    state=state,
                    sample=sample,
                    output=output_prefix,
                )
        else:
            raise RuntimeError(
                f'the signature of analysis "{name}" is not supported')


cli.add_command(analyze)


@click.command()
@click.argument('gene-list', nargs=-1)
def genes(gene_list):
    def _analysis(state, sample, output):
        analyze_genes(
            state.histonet,
            state.std,
            sample,
            gene_list,
            output_prefix=output,
            device=DEVICE,
        )
    return 'gene list', _analysis


analyze.add_command(genes)


@click.command()
@click.argument('gene-list', nargs=-1)
@click.option('--factor', type=int, multiple=True)
@click.option('--truncate', type=int, default=25)
@click.option('--regex/--no-regex', default=True)
def gene_profiles(gene_list, factor, truncate, regex):
    def _analysis(state, samples, output):
        analyze_gene_profiles(
            std=state.std,
            genes=list(gene_list),
            factors=factor if len(factor) > 0 else None,
            truncate=truncate,
            regex=regex,
            output_prefix=output,
        )
    return 'gene profiles', _analysis


analyze.add_command(gene_profiles)


@click.command()
def default():
    def _analysis(state, sample, output):
        default_analysis(
            state.histonet,
            state.std,
            sample,
            output_prefix=output,
            device=DEVICE,
        )
    return 'default', _analysis


analyze.add_command(default)


@click.command()
@click.argument(
    'regions-file',
    metavar='regions',
    type=click.File('rb'),
)
def impute(regions_file):
    regions = pd.read_csv(regions_file)

    regions_dir = os.path.dirname(regions_file.name)

    def _path(p):
        return (
            p
            if os.path.isabs(p) else
            os.path.join(regions_dir, p)
        )

    if 'regions' not in regions.columns:
        raise ValueError('regions file must contain a "regions" column')

    def _analysis(state, samples, output, **_):
        nonlocal regions

        if 'name' in regions.columns:
            samples_dict = {s.name: s for s in samples}

            def _sample(n):
                if n not in samples_dict:
                    raise ValueError(f'name "{n}" is not in the design file')
                return samples_dict[n]

            samples = [*map(_sample, regions.name)]
        else:
            if len(regions) != len(samples):
                raise ValueError(
                    'if the regions file does not contain a "name" column, '
                    'it must have the same length as the design file.'
                )

        regions = [
            Image.new_from_file(os.path.join(regions_dir, r))
            for r in regions.regions
        ]

        for sample, region in zip(samples, regions):
            means, samples, index = impute_counts(
                state.histonet,
                state.std,
                sample,
                region,
                device=DEVICE,
            )
            os.makedirs(os.path.join(output, sample.name))
            (
                pd.DataFrame(
                    means.mean(0).numpy(),
                    index=pd.Index(index, name='n'),
                    columns=state.std.genes,
                )
                .to_csv(os.path.join(output, sample.name, 'imputed.csv.gz'))
            )
            (
                pd.concat(
                    [
                        pd.DataFrame(
                            s.numpy().astype(int),
                            index=pd.Index(index, name='n'),
                            columns=state.std.genes,
                        )
                        for s in samples
                    ],
                    keys=list(range(len(samples))),
                    names=['sample'],
                )
                .to_csv(os.path.join(output, sample.name, 'samples.csv.gz'))
            )
    return 'imputation', _analysis


analyze.add_command(impute)


@click.command()
@click.argument(
    'regions-file',
    metavar='regions',
    type=click.File('rb'),
)
@click.option('--normalize/--no-normalize', default=True)
@click.option('--trials', type=int, default=100)
def dge(regions_file, normalize, trials):
    regions = pd.read_csv(regions_file)

    regions_dir = os.path.dirname(regions_file.name)

    def _path(p):
        return (
            p
            if os.path.isabs(p) else
            os.path.join(regions_dir, p)
        )

    if 'regions' not in regions.columns:
        raise ValueError('regions file must contain a "regions" column')

    def _analysis(state, samples, output, **_):
        nonlocal regions

        if 'name' in regions.columns:
            samples_dict = {s.name: s for s in samples}

            def _sample(n):
                if n not in samples_dict:
                    raise ValueError(f'name "{n}" is not in the design file')
                return samples_dict[n]

            samples = [*map(_sample, regions.name)]
        else:
            if len(regions) != len(samples):
                raise ValueError(
                    'if the regions file does not contain a "name" column, '
                    'it must have the same length as the design file.'
                )

        regions = [
            Image.new_from_file(os.path.join(regions_dir, r))
            for r in regions.regions
        ]

        dge_analysis(
            state.histonet,
            state.std,
            samples=samples,
            regions=regions,
            output=output,
            normalize=normalize,
            trials=trials,
            device=DEVICE,
        )

    return 'differential gene expression', _analysis


analyze.add_command(dge)


if __name__ == '__main__':
    cli()
