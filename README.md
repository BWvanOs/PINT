![PINT banner](Screenshots/Logo.png)

# PINT

**Current version:** 0.6.1

PINT is an IMC/CyTOF viewer for image loading, normalization, mask visualization, and neighborhood analysis.

## Installation

### 1. Make sure Git is installed

```bash
Windows:
conda install -c conda-forge git

Linux:
sudo apt install git / brew install git

MacOS:
brew install git
```

### 2. Clone the repo from github and install
```bash
git clone https://github.com/BWvanOs/PINT.git
```

Go to the folder you downloaded pint into and install it:

```bash
cd PINT
conda env create -f environment.yml
conda activate pint
pint

```

### 3. Install an update or a specific version 
To update to the newest version on the main branch:
```bash
cd /path/to/PINT
git checkout main
git pull
conda activate pint
conda env update -n pint -f environment.yml --prune
```

Download a specific version (e.g. version 0.6.1)
```bash
git clone --branch v0.6.1 https://github.com/BWvanOs/PINT.git
cd PINT
conda env create -f environment.yml
conda activate pint
pint
```

Or rollback to an older version if the new one is giving problems
```bash
cd /path/to/PINT
git fetch --tags
git checkout v0.6.0
conda activate pint
conda env update -n pint -f environment.yml --prune
pint
```

List all versions of PINT:
```bash
cd /path/to/PINT
git fetch --tags
git checkout v0.6.0
conda activate pint
conda env update -n pint -f environment.yml --prune
pint
```



## Removing outlier, thresholding, transformation, normalization


