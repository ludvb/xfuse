#!/usr/bin/env python3

""" refresh_version.py

writes dloss/__version__ to correspond to the current git version
"""

import re

import os.path as osp

import subprocess as sp


def _output_from(cmd):
    return (
        sp.run(
            cmd.split(' '),
            capture_output=True,
            check=True,
        )
        .stdout
        .decode()
        .strip()
    )


try:
    VERSION = _output_from('git describe --dirty')
    # Modify according to PEP440
    VERSION = (
        re.compile(r'([^\-]+?)-([^\-]+?)-(.+?)')
        .sub(r'\g<1>.post\g<2>+\g<3>', VERSION)
        .replace('-', '.')
    )
except sp.CalledProcessError:
    try:
        VERSION = _output_from('git describe --dirty --always')
        VERSION = '0.0.0+untagged.{}'.format(VERSION.replace('-', '.'))
    except sp.CalledProcessError:
        VERSION = '0.0.0+git.error'
except FileNotFoundError:
    VERSION = '0.0.0+no.git'

try:
    diff = _output_from('git diff')
except (FileNotFoundError, sp.CalledProcessError):
    diff = ''

with open(
    osp.join(
        osp.dirname(__file__),
        'dloss',
        '__version__.py',
    ),
    'w',
) as f:
    f.write(''.join(map(lambda x: x + '\n', [
        '""" Generated automatically---don\'t edit!',
        '"""',
        '',
        f'__version__ = \'{VERSION:s}\'',
        '__diff__ = """' + diff.replace('"', '\\"') + '"""',
    ])))