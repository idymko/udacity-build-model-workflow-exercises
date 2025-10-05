# Instructions
In this exercise you will experiment with different ways of deploying the exported model for
online and offline inference.

# Preliminary step
First we need to fetch the production model. We are going to save it into the ``model``
directory:

```bash
# don't forget to activate the udacity env where wandb is installed
> exercise_16 % conda activate udacity
> exercise_16 % wandb artifact get genre_classification_prod/model_export:prod --root model
```

This will create a folder called "model" with the following files:
```bash
> model % ls
MLmodel			input_example.json	python_env.yaml
conda.yaml		model.pkl		requirements.txt
```

# Offline (batch) inference
Let's run inference on the test set. 

## Steps
1. Use the W&B CLI to download the ``genre_classification_prod/data_test.csv:latest`` artifact
   locally (``wandb artifact get``...)

```bash
> exercise_16 % wandb artifact get genre_classification_prod/data_test.csv:latest
```

This will create a folder "artifacts" with "data_test.csv" inside.
   
2. Use ``mlflow models predict ...`` to perform batch inference on it
   > Remember to use ``-m model`` to specify the directory where you stored the artifact

```bash
mlflow models predict \
               -t csv \
               -i artifacts/data_test.csv:v0/data_test.csv \
               -m model
```

> NOTE: in case of errors, you may need to install `pyenv` and `virtualenv`:
```bash
brew update
brew install pyenv
pip install virtualenv
```

# Online inference
Here we use the model for online prediction exposing the model as a REST API.

1. Let's start the API using ``mlflow models serve ...`` 
   > HINT: remember that we saved the model into the ``model`` directory
   The API is now ready to perform inference.

```bash
mlflow models serve -m model
# run in the background
mlflow models serve -m model & 
```
This will provide the API with e.g. `<PID>`= [32078].
```bash
[2025-10-05 12:41:36 +0200] [32078] [INFO] Listening at: http://127.0.0.1:5000 (32078)
```

2. Open Jupyter in another terminal (simply type `jupyter notebook`), and in a notebook use the ``requests`` library
   to interrogate the API and do inference on the provided ``data_sample.json``:
   ```python
   import requests
   import json

   with open("starter/data_sample.json") as fp:
    data = json.load(fp)

   # adapt the JSON payload to the expected format for MLflow > 2.0.
   formatted_data = {
      "dataframe_split": {
         "columns": data["columns"],
         "data": data["data"]
      }
   }
   # required format
   #{
   #    "dataframe_split": {
   #        "columns": [...],
   #        "data": [...]
   #    }
   #}
   results = requests.post("http://127.0.0.1:5000/invocations", json=formatted_data)

   # Print the response
   if results.status_code == 200:
      print("Prediction:", results.json())
   else:
      print("Error:", results.status_code, results.json())
   ```

* To exit the job press `Ctrl+C`.
* To stop the process use the command `kill <PID>`.
* `<PID>` is mentioned above or can be found by running `lsof -i :5000`. 
   
## Bonus: docker deployment
You can also use docker to build an image and then deploy to a Cloud provider 
(AWS, GCP, Azure...). Expose port 5000 of that machine to the world, and you will be able to
use your model from whenever as a simple API call.

1. Create the docker image:
   ```bash
   mlflow models build-docker -m model -n "genre_classification"
   ```
   This will take a few minutes (of course, you need docker installed)

2. Follow the procedure for your Cloud provider of choice to deploy a Docker image

3. Open port 5000 on the machine hosting the image

4. Use requests to interrogate that machine, by using the snippet we used earlier and substituting
   ``http://localhost:5000/invocations`` with ``[url to the deployed machine]:5000/invocations``