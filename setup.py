#!/usr/bin/env python

from setuptools import setup, find_packages

required = [
    'flax>=0.4.1',
    'jax==0.4.13',
    'jaxtyping>=0.2.20',
    'jaxlib==0.4.13',
    'pytest==7.4.0',
    'matplotlib>=3.5.1',
    'numpy>=1.22.2',
    'optax>=0.1.1',
    'scipy>=1.8.0',
    'wandb>=0.12.11',
    'distrax~=0.1.2',
    'argparse-dataclass>=0.2.1',
    'jaxutils',
    'chex',
    'brax @ git+https://git@github.com/lenarttreven/brax.git',
    'trajax @ git+https://git@github.com/lenarttreven/trajax.git',
    # 'bsm @ git+https://git@github.com/lasgroup/bayesian_statistical_models.git',
]

extras = {}
setup(
    name='bsm',
    version='0.0.1',
    license="MIT",
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=required,
    extras_require=extras,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.11",
    ],
)
