# Instructions
In this exercise you will export a model using the parameters we have found in the previous exercise 
during the experimentation phase.

Within the ``random_forest/run.py`` step complete the ``export_model`` function. Instructions are
provided there.

Once you are done, execute the pipeline setting ``random_forest_pipeline.random_forest.max_depth``
to 13 and ``random_forest_pipeline.tfidf.max_features`` to 10. These parameters are almost the best
performing, and give a small model which is going to be very fast in production. After the run 
your exported pipeline will be saved as the artifact ``exercise_12/model_export``.

```bash
# go to folder
cd /Users/dkysylychyn/python/MLDevOps/udacity-build-model-workflow-exercises/lesson-4-training-validation-experiment-tracking/exercises/exercise_12/solution_dk

# run mlflow
mlflow run . -P hydra_options="-m random_forest_pipeline.random_forest.max_depth=13 random_forest_pipeline.tfidf.max_features=10"

# or set these values directly in config.yaml file
mlflow run .
```