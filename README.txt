##Welcome to PINT, a normalization and noise removal tool for Imaging mass cytometry.

##Prerequisites:

- Git
- Conda (Miniconda (or if prefered Anaconda))

##Installation (recommended)

###First clone the repository
    git clone https://github.com/BWvanOs/PINT
    cd PINT-main

###Create conda environment
    conda env create -f environment.yml
    conda activate pint_env
    pip install -e .
    PINT

##For daily use after installation:
    conda activate pint_env
    PINT
