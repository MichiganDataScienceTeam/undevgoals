# undevgoals
United Nations Millennium Development Goals Competition: https://www.drivendata.org/competitions/1/united-nations-millennium-development-goals/

## Setup

1. Install `virtualenv` if you haven't already.

```
pip install --upgrade virtualenv
```

2. Create a new `virtualenv` with the correct dependencies:

```
virtualenv undg -p python3
source undg/bin/activate
pip install -r requirements.txt
```

3. Place data in `./data/`. There should be two files:

 - TrainingSet.csv
 - SubmissionRows.csv

4. Run the training code

```
python train.py
```
