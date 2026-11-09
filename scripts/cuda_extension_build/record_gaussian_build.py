# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0

"""Include the native binary and its build inputs in the same wheel."""

import json
import os
from pathlib import Path

from setuptools import Command
from setuptools.errors import SetupError

from cuda_extension_build.inputs import COMPILER_FLAGS, PACKAGES, RECORD, build_inputs, sha256


class record_gaussian_build(Command):
    description = "record Gaussian extension build inputs and native binary hashes"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        inputs = self.distribution.real2sim_build_inputs
        if inputs != build_inputs(Path.cwd().parents[3]):
            raise SetupError("Gaussian extension build inputs changed during compilation")
        package = PACKAGES[self.distribution.get_name().replace("_", "-")]
        build_lib = Path(self.get_finalized_command("build").build_lib)
        binaries = {str(path.relative_to(build_lib)): sha256(path)
                    for path in (build_lib / package).glob("*.so")}
        if not binaries:
            raise SetupError(f"no native binary was built for {package}")
        (build_lib / package / RECORD).write_text(
            json.dumps({"inputs": inputs, "binaries": binaries,
                        "compiler_flags": {key: os.environ.get(key, "") for key in COMPILER_FLAGS}},
                       sort_keys=True) + "\n"
        )
