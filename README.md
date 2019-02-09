# undevgoals
United Nations Millennium Development Goals Competition: https://www.drivendata.org/competitions/1/united-nations-millennium-development-goals/

## Setup

1. Install `conda` if you haven't already.

2. Create a new `conda` environment with the correct dependencies:

```
conda env create -f environment.yml
source activate undg
```

3. Place data in `./data/`. There should be two files:

 - TrainingSet.csv
 - SubmissionRows.csv

4. Run the training code

```
python train.py
```
