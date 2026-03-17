Current version 0.5

##Install like this:

##Make sure git is installed:

##conda install -c conda-forge git    # or: sudo apt install git / brew install git

##clone the repo:
##git clone https://github.com/<OWNER>/<REPO>.git

##cd PINT

##conda env create -f environment.yml
##conda activate pint
##python PINT.py

##Install update
##one way is to pull new version:
    ##cd /path/to/PINT
    ##git pull

##Easierst way (also if environment changed)
    ##conda activate pint
    ##conda env update -n pint -f environment.yml --prune
