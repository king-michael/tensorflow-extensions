# tensorflow-extensions
Extensions around Tensorflow


## dataset
Various Version around Tensorflow Datsets.
Check out [https://github.com/king-michael/numpy-extensions](https://github.com/king-michael/numpy-extensions) for more numpy options.


* `create_dataset_from_listnpy(npy_files, num_features=None, dtype = tf.float32)` <br>
    Creates a TFDataset from a list of npy files.

* `NumpyDataset(npyfiles, dtype=None, verbose=False)` <br>
    Class that implements a numpy dataset 
    * `get_sample(self, percentage=None, n_samples=None, size=None, replace=False)` <br>
        Get a sample of the dataset. <br>
        Only of the options `['percentage', 'n_samples', 'size']` is usable at one time.

### utils
Utility functions      
* ` get_batch(array, batchsize, axis=0)` <br>
    Creates a generator for an array, returning a batch along the first axis.
        
* `get_a_random_sample_from_list(list_data, percentage=None, n_samples=None, dtype=np.float64, replace=True)` <br>
    Draw a random sample from a list of data.
    
