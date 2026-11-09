# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Validate the inputs before compiling a Gaussian CUDA extension."""

from pathlib import Path
import subprocess

from setuptools import Command
from setuptools.errors import SetupError

from cuda_extension_build.inputs import build_inputs


class check_gaussian_wrapping(Command):  # distutils resolves the class by its command name
    description = "validate GaussianWrapping source and build toolchain"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        root = Path.cwd().parents[3]
        try:
            self.distribution.real2sim_build_inputs = build_inputs(root, building=True)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            raise SetupError(str(exc)) from exc
