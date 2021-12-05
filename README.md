
Pull tensorflow docker container:

    docker pull tensorflow/tensorflow 

Run:

    docker run -it --rm -v $PWD:/tmp -w /tmp tensorflow/tensorflow python ./path/to/script.py