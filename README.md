# Transfer Learning 101

### How do I get set up? ###

1. Make sure you either:
  -- create a virtual environment
  -- or are running a docker container with docker ([see here](https//hub.docker.com/r/tensorflow/tensorflow/))
2. Install the dependencies with the `requirements.txt` in this repo.

Quick start:
```sh
$ virtualenv -p $(which python3) py-tf
$ source py-tf/bin/activate
$ pip install -r requirements.txt
```

Both Python 2.7 and Python 3.5 should work (tested with 2.7 and 3.5). That said, it's time to let 2.7 go.

### Finding your way ###
`sh` is your entry point; there's a bash script to showcase the usage of the retraining code

`scripts` contains the retraining code; notice there have been some modifications to the original Tensorflow github script, 
but the interface is the same.

`tf_files` Just like in the original Tensorflow Transfer Learning [tutorial](https://www.tensorflow.org/tutorials/image_retraining): data, models and logs will be saved here. 

#### Tensorboard ####
Notice tf_files/training_summaries contains the following structure:
   * architecture
      * training_timestamp
        * train
        * validation

This way you will be able to compare the loss function and other metrics from different trainings in the same Tensoboard
plot.

![tensorboard](doc/tensorboard_multiple.png)

### License ###
I'm redistributing modifications under the same Apache 2.0 license from the original Tensorflow code.
