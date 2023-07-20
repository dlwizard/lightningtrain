#!/usr/bin/env python
from setuptools import find_packages, setup
# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="lightningtrain",
    python_requires='>=3.10.6',
    version="0.0.4",
    description="PyTorch Lightning Project Setup",
    author="Santu Hazra",
    author_email="ec.santuh@gmail.com",
    url="https://github.com/dlwizard/lightningtrain.git",
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "lightningtrain_train = lightningtrain.train:main",
            "lightningtrain_eval = lightningtrain.eval:main",
            "lightningtrain_infer = lightningtrain.infer:main"
        ]
    },
)
