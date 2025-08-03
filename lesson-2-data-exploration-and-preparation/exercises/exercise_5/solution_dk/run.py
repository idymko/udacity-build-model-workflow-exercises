#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    
    logger.info("Creating run exercise_5")
    run = wandb.init(project="exercise_5", job_type="process_data")

    logger.info(f"Download the input artifact {args.input_artifact}")
    artifact = run.use_artifact(args.input_artifact)
    local_path = artifact.file()
    
    logger.info(f"Open artifact from local path {local_path} with pandas")
    df = pd.read_parquet(local_path)
    
    logger.info("Drop duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    
    logger.info("Remove NAs and add a new feature")
    df['title'] = df['title'].fillna(value='')              # replace NA with empy char
    df['song_name'] = df['song_name'].fillna(value='')      # replace NA with empy char
    df['text_feature'] = df['title'] + ' ' + df['song_name']      # concat 'title' and 'song_name' to a new feature 
    ## NOTE: again, in a real setting, you will have to make sure that your 
    ## feature store provides this text_feature at inference time, OR, you 
    ## will have to move the computation of this feature to the inference pipeline.
    
    # Save the result to a file and 
    logger.info(f"Save cleaned dataframe to a file {args.artifact_name}")
    df.to_csv(args.artifact_name, index=False)
    
    logger.info(f"Upload cleaned dataframe to an artifact {args.artifact_name}")
    clean_artifact = wandb.Artifact(
                name=args.artifact_name,
                type=args.artifact_type,
                description=args.artifact_description
            )
    clean_artifact.add_file(args.artifact_name)
    
    logger.info("Logging artifact")
    run.log_artifact(clean_artifact)
    
    logger.info("Finish the run")
    run.finish() # not necessary - run will automatically finish when 
                 #                  script completes if only one run is used within the script


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)

    #see README.md on how to run the project