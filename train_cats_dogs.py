import os
import re
import time

import mlflow
import pandas as pd
import pytorch_lightning as pl
import torch
from loguru import logger
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader

from datasets.cats_vs_dogs import CatsDogsDataset, get_preprocessor
from modules.cats_dogs_classifier import CatsDogsClassifier


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    logger.info(f"run_id: {r.info.run_id}")
    logger.info(f"artifacts: {artifacts}")
    logger.info(f"params: {r.data.params}")
    logger.info(f"metrics: {r.data.metrics}")
    logger.info(f"tags: {tags}")


def get_dvc_rev(dvc_fp):
    with open(dvc_fp) as f:
        s = f.read()
        revs = re.findall(r"rev: (\S+)", s)
    return revs[0] if revs else ""


if __name__ == "__main__":
    # Set random seeds for reproducibility purpose
    seed = 42
    pl.seed_everything(seed=seed, workers=True)

    # Create an experiment. By default, if not specified, the "default" experiment is used. It is recommended to not use
    # the default experiment and explicitly set up your own for better readability and tracking experience.
    client = MlflowClient()
    experiment_name = "Cats vs dogs classification"
    model_architecture = "mobilenetv2"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = f"cats_vs_dogs_pt_cnn_{timestamp}"

    run_name = model_name
    try:
        experiment_id = client.create_experiment(experiment_name)
        experiment = client.get_experiment(experiment_id)
    except MlflowException:
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id

    # Fetch experiment metadata information
    logger.info(f"Name: {experiment.name}")
    logger.info(f"Experiment_id: {experiment.experiment_id}")
    logger.info(f"Artifact Location: {experiment.artifact_location}")
    logger.info(f"Tags: {experiment.tags}")
    logger.info(f"Lifecycle_stage: {experiment.lifecycle_stage}")

    training_output_dir = os.path.join("./experiments/training_outputs", model_name)
    checkpoints_dir = os.path.join(training_output_dir, "checkpoints")

    dataset_dvc_fp = "cats_vs_dogs.dvc"
    dataset_version = get_dvc_rev(dataset_dvc_fp)

    use_imagenet_pretrained_weights = True

    gpus = 1 if torch.cuda.is_available() else 0

    params = {
        "model": model_architecture,
        "dataset_version": dataset_version,
        "seed": seed,
        "batch_size": 32,
        "num_workers": 8,
        "gpus": gpus,
        "precision": 32,
        "max_epochs": 25,
        "lr": 1e-1,
        "use_imagenet_pretrained_weights": use_imagenet_pretrained_weights,
        "early_stopping_patience": 3,
    }

    # initialize the data set splits
    df_train = pd.read_csv("data_splits/train.csv")
    dataset_train = CatsDogsDataset(
        df_train["img_fp"].values,
        df_train["is_dog"].values,
        preprocess=get_preprocessor(imagenet=use_imagenet_pretrained_weights),
    )
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params["num_workers"],
    )

    df_validation = pd.read_csv("data_splits/validation.csv")
    dataset_validation = CatsDogsDataset(
        df_validation["img_fp"].values,
        df_validation["is_dog"].values,
        preprocess=get_preprocessor(imagenet=use_imagenet_pretrained_weights),
    )
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    params["train_size"] = len(dataset_train)
    params["validation_size"] = len(dataset_validation)

    # model
    model = CatsDogsClassifier(
        imagenet_weights=use_imagenet_pretrained_weights, lr=params["lr"]
    )
    monitor = "val_accuracy"
    mode = "max"
    checkpoint_name_format = "{epoch:03d}_{" + monitor + ":.3f}"

    callbacks = [
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            dirpath=checkpoints_dir,
            filename=checkpoint_name_format,
            monitor=monitor,
            save_last=True,
            save_top_k=-1,  # save all checkpoints
            mode=mode,
            every_n_epochs=1,
        ),
        pl.callbacks.early_stopping.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=params["early_stopping_patience"],
            verbose=True,
        ),
    ]

    tensorboard_logger = pl_loggers.TensorBoardLogger(
        training_output_dir, name="tensorboard"
    )

    trainer = pl.Trainer(
        gpus=params["gpus"],
        precision=params["precision"],
        max_epochs=params["max_epochs"],
        callbacks=callbacks,
        deterministic=True,
        logger=tensorboard_logger,
    )

    # Activate auto logging for pytorch lightning module
    mlflow.pytorch.autolog()

    # Launch training phase
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        logger.info("tracking uri:", mlflow.get_tracking_uri())
        logger.info("artifact uri:", mlflow.get_artifact_uri())
        logger.info("start training")

        # log training parameters
        mlflow.log_params(params)

        # save dataset's dvc file
        mlflow.log_artifact(dataset_dvc_fp)

        trainer.fit(model, dataloader_train, dataloader_validation)
        mlflow.log_artifacts(training_output_dir)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
