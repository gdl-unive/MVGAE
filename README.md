# Mixture of Variational Graph Autoencoders

[![Docker Pulls](https://img.shields.io/docker/pulls/ripper346/landmark-variational-autoencoder?logo=docker&label=MVGAE%20pulls)](https://hub.docker.com/r/ripper346/mvgae)

[![Docker Image Size (tag)](https://img.shields.io/docker/image-size/ripper346/mvgae/cuda?logo=docker&label=MVGAE-cuda)](https://hub.docker.com/repository/docker/ripper346/mvgae/tags)
[![Docker Image Size (tag)](https://img.shields.io/docker/image-size/ripper346/mvgae/cpu?logo=docker&label=MVGAE-cpu)](https://hub.docker.com/repository/docker/ripper346/mvgae/tags)

## Requirements
The repository is made to run in a Docker container, we provide the images from
Docker Hub or you can compile the Dockerfile.

There is also a Docker compose file you can run. At the moment it links to an
image from Docker Hub but it can be replaced with a link to the Dockerfile.

The docker image uses Conda, so if you want to run the code straight from your
machine you can get the various commands to execute from the Dockerfile.

### How to run
```bash
python3 ./train.py
```

For the parameters refer to the `train.py` file and `config_XX.json` files.
