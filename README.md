# tensorflow-extensions
Extensions around Tensorflow


## dataset
Various Version around Tensorflow Datsets.
Check out [https://github.com/king-michael/numpy-extensions](https://github.com/king-michael/numpy-extensions) for more numpy options.

* `NumpyDataset(npyfiles, dtype=None, verbose=False)` <br>
    Class that implements a numpy dataset 
    * `get_sample(self, percentage=None, n_samples=None, size=None, replace=False)` <br>
        Get a sample of the dataset. <br>
        Only of the options `['percentage', 'n_samples', 'size']` is usable at one time.