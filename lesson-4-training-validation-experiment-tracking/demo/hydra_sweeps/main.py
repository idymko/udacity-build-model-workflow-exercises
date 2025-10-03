import mlflow
import os
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "component"),
        "main",
        parameters={
            "a": config["parameters"]["a"],
            "b": config["parameters"]["b"],
        },
    )


if __name__ == "__main__":
    go()

    """
    If we want Hydra to do a sweep, we need to specify the values we want to explore and then the options -m ("multi-run"). So for example:
    > mlflow run . -P hydra_options="-m parameters.a=3,4,5"
    will generate 3 runs. The first one will have a=3, the second a=4 and the third a=5.

    """