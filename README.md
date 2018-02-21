Machine Learning in Practice
============================

Repository for the [Dog Breed
Identification](http://www.kaggle.com/c/dog-breed-identification) Kaggle
competition. Provided a strictly canine subset of
[ImageNet](https://www.kaggle.com/c/imagenet-object-detection-challenge), we
create a classification system to correctly classify breeds of dogs.

### Data
[Download here](https://www.kaggle.com/c/dog-breed-identification/data). Make
sure you download and extract the data in a folder called `data` (this folder
is in `.gitignore` so the data does not have to reside on the repository).

    data
    ├── labels.csv
    ├── sample_submission.csv
    ├── test
    └── train

### System requirements
- [Python >3.6](https://www.python.org/downloads/release/python-364/)
- keras (will install numpy and scipy as well)
- sklearn

#### Virtual environment
Ideally you work in a Python virtual environment. If you don't know how to set
this up, here are some instructions.

To create a new virtual environment, path can be anything you choose (for
example `/home/koen/venvs/mlip`):

    $ python3 -m venv <PATH>

Activate virtual environment:

    $ source <PATH>/bin/activate

Once activated, anything you install using `pip` is installed in the virtual
environment separately from your system Python. Keras and sklearn can be
installed using `pip`:

    $ pip install keras sklearn
