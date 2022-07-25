# Cats vs dogs classification: effortlessly track your model experiments with DVC and MLflow

## Introduction

This repository is an introductory example of MLflow and DVC usage for model experiment tracking.

This example is referred by the series of article : TODO link to articles

This repository only has a minimalist code base for starting to play with MLflow and DVC tools. Further instructions and
good practices about how to manage your model experiment runs are explained in the articles.

The classic "Cats vs Dogs" classification challenge is used to illustrate.

- DVC is used for versioning different "Cats vs dogs" data set versions.
- MLflow is used for model artifacts, hyper-parameters and metrics tracking.

## Prerequisites

### Dependencies

#### Create virtual environment

    conda create -n "mlflow_dvc_tuto" python=3.9

#### Install Pytorch dependencies

Follow the instructions for installing Pytorch [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
Preferably install pytorch 1.8.2 (LTS) with pip and use CUDA compute platform if you have a compatible GPU.

#### Install dependencies

    conda activate mlflow_dvc_tuto
    pip install -r requirements.txt

### Set up a DVC data set registry for the "cats vs dogs" data set

Please follow the instructions in the related article (TODO link) on DVC and how to set up a data set registry.

### Set up MLflow tracking server

You can find further information about MLflow with:
- [MLflow documentation](https://www.mlflow.org/docs/latest/index.html)
- The article that overviews MLflow TODO article link

#### Locally in the Git repo

If you want to start experimenting right away without additional setup, you can just use the training
script `train_cats_dogs.py` as is.

The MLflow experiments runs will be saved in `./mlruns` directory.

You will be able to start tracking your experiments with MLflow UI:

    mlflow ui

#### Remote server and cloud-providers support for MLflow

Set up a new tracking method for MLflow by checking the
documentation [MLflow tracking](https://www.mlflow.org/docs/latest/tracking.html)

Assuming you already have a remote MLflow server setup or you use a cloud-provider MLflow support (for instance Azure ML
experiment), with all the credentials to access it (read and write access), just set up your new tracking URI
with [mlflow.set_tracking_uri()](https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri) in
the code.

For instance, if you have setup a remote MLflow server at `<hostname>:8888`, set up your new tracking URI before the
start of a new experiment run:

    mlflow.set_tracking_uri("<hostname>:8888")

You will be able to start tracking your experiment with MLflow UI by directly accessing `<hostname>:8888` in a browser.
