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
is in `.gitignore` so the data does not have to reside on the repository)
inside where you cloned this repository.

    this/repository/data
    ├── labels.csv
    ├── sample_submission.csv
    ├── test
    └── train

### System requirements
- [Python >3.6](https://www.python.org/downloads/release/python-364/)
- keras (will install numpy and scipy as well)
- sklearn (machine learning package)
- matplotlib (visualization)
- tensorflow-gpu (keras backend; you can also use regular tensorflow)
- pandas (easy data inspection)
- tqdm (progress bar)
- PIL (image library)
- h5py (HDF5 binary data format)
- jupyter (optional, to work with notebooks)
- seaborn (optional, to plot confusion matrices)

#### Virtual environment
Ideally you work in a Python virtual environment. If you don't know how to set
this up, here are some instructions.

To create a new virtual environment, path can be anything you choose (for
example `/home/koen/venvs/mlip`):

    $ python3 -m venv <PATH>

Activate virtual environment:

    $ source <PATH>/bin/activate

Once activated, anything you install using `pip` is installed in the virtual
environment separately from your system Python:

    $ pip install keras sklearn matplotlib tensorflow-gpu pandas tqdm pillow h5py jupyter

### Code formatting
Code should be formatted according to
[PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines. 4 spaces
indentation etc. :)

### TensorBoard
`train.py` logs training progress to the `./training_log` directory; these logs
can be visualized using `tensorboard` (example command from within project
directory).

    $ tensorboard --logdir=./training_log
