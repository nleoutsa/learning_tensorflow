
Pull [tensorflow](https://www.tensorflow.org/) docker container:

    $ docker pull tensorflow/tensorflow 

Build a new image (additional libraries installed via Dockerfile):

    $ docker build -t tensorflow-custom . 

Run:

    $ docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow-custom python ./path/to/script.py

Datasets can be found at [Kaggle](https://www.kaggle.com/datasets)