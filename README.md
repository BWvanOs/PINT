![PINT banner](Screenshots/)

# PINT

**Current version:** 0.6

PINT is an IMC/CyTOF viewer for image loading, normalization, mask visualization, and neighborhood analysis.

## Installation

### 1. Make sure Git is installed

```bash
conda install -c conda-forge git

or on Linux

sudo apt install git / brew install git
```

### 2. Clone the repo from github and install
```
git clone https://github.com/BWvanOs/PINT.git
```

#### Go to the folder you downloaded pint into and install it:

```
cd PINT

conda env create -f environment.yml
conda activate pint
pint

```

### 3. To update either CD into the folder and pull the new version
```
cd /path/to/PINT
git pull

```

Or if the environment changed just pull the new version
```
conda activate pint
conda env update -n pint -f environment.yml --prune
```

## Removing outlier, thresholding, transformation, normalization


