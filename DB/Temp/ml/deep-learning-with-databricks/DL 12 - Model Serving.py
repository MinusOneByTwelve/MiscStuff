# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Serving
# MAGIC 
# MAGIC There are many deployment options for machine learning models.  This notebook explores a more complex deployment scenario involving the real time deployment of a convolutional neural network using REST and Databricks MLflow Model Serving.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Create a **`pyfunc`** to serve a **`keras`** model with pre and post processing logic
# MAGIC - Save the **`pyfunc`** for downstream consumption 
# MAGIC - Serve the model using a REST endpoint
# MAGIC 
# MAGIC **NOTE:** *You need <a href="https://docs.databricks.com/applications/mlflow/model-serving.html#requirements" target="_blank">cluster creation</a> permissions to create a model serving endpoint. The instructor will either demo this notebook or enable cluster creation permission for the students from the Admin console.*

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## Model Serving in Databricks
# MAGIC 
# MAGIC The MLflow model registry in Databricks is now integrated with MLflow Model Serving.  This is currently intended for development use cases and is therefore not intended for production.  In this module, you will create a wrapper class around a **`keras`** model that provides custom pre and post processing logic necessary for this more complex deployment scenario. 
# MAGIC 
# MAGIC For additional background, see the following resources:
# MAGIC 
# MAGIC - <a href="https://databricks.com/blog/2020/06/25/announcing-mlflow-model-serving-on-databricks.html" target="_blank">Databricks blog on model serving</a>
# MAGIC - <a href="https://github.com/mlflow/mlflow/tree/master/examples/flower_classifier" target="_blank">Example of an image classifier</a>
# MAGIC - <a href="https://www.mlflow.org/docs/latest/models.html#example-saving-an-xgboost-model-in-mlflow-format" target="_blank">Example of a custom loader used with XGBoost</a>

# COMMAND ----------

# MAGIC %md ## Creating a Wrapper Class using **`pyfunc`**

# COMMAND ----------

# MAGIC %md Create a **`keras`** model using a reference architecture and pretrained weights.

# COMMAND ----------

import tensorflow as tf
tf.random.set_seed(42)

model = tf.keras.applications.VGG16(weights="imagenet")
model.summary()

# COMMAND ----------

# MAGIC %md Create a small dataset to test the model.  This is two images of cats.

# COMMAND ----------

import pandas as pd
import base64

filenames = [f"{DA.paths.datasets}/img/cats/cats2.jpg".replace("dbfs:/", "/dbfs/"), 
             f"{DA.paths.datasets}/img/cats/cats4.jpg".replace("dbfs:/", "/dbfs/")]

def read_image(path: str) -> bytes:
    """Reads an image from a path and returns the contents in bytes"""
    with open(path, "rb") as f:
        image_bytes = f.read()
    return image_bytes

data = pd.DataFrame(data=[base64.encodebytes(read_image(x)) for x in filenames], columns=["image"])
data

# COMMAND ----------

# MAGIC %md Save the model using **`mlflow`**.  

# COMMAND ----------

import mlflow
import mlflow.keras

suffix = DA.unique_name("-")
model_name = f"keras-model_{suffix}"

with mlflow.start_run() as run:
    mlflow.keras.log_model(artifact_path=model_name, keras_model=model)
    model_uri = f"runs:/{run.info.run_id}/{model_name}"

print(f"Model saved to {model_uri}")

# COMMAND ----------

# MAGIC %md Create a wrapper class that includes the following as a **`pyfunc`**:
# MAGIC 
# MAGIC - A **`load_context`** method to load in the model. 
# MAGIC - Custom featurization logic that parses base64 encoded images (necessary for HTTP requests)
# MAGIC - Custom prediction logic that reports the top class and its probability

# COMMAND ----------

import mlflow

class KerasImageClassifierPyfunc(mlflow.pyfunc.PythonModel):
  
    def __init__(self):
        self.model = None
        self.img_height = 224
        self.img_width = 224
    
    def load_context(self, context=None, path=None):
        """
        When loading a pyfunc, this method runs automatically with the related
        context.  This method is designed to load the keras model from a path 
        if it is running in a notebook or use the artifact from the context
        if it is loaded with mlflow.pyfunc.load_model()
        """
        import numpy as np
        import tensorflow as tf

        if context: # This block executes for server run
            model_path = context.artifacts["keras_model"]
        else: # This block executes for notebook run
            model_path = path

        self.model = mlflow.keras.load_model(model_path)
    
    def predict_from_bytes(self, image_bytes):
        """
        Applied across numpy representations of the model input, this method
        uses the appropriate decoding based upon whether it is run in the 
        notebook or on a server
        """
        import base64

        try: # This block executes for notebook run
            image_bytes_decoded = base64.decodebytes(image_bytes)
            img_array = tf.image.decode_image(image_bytes_decoded)
        except: # This block executes for server run
            img_array = tf.image.decode_image(image_bytes) 

        img_array = tf.image.resize(img_array, (self.img_height, self.img_width))
        img_array = tf.expand_dims(img_array, 0)
        prediction = self.model.predict(img_array)
        return prediction[0]
  
    def postprocess_raw_predictions(self, raw_prediction):
        """
        Post processing logic to render predictions in a human readable form
        """
        from tensorflow.keras.applications.vgg16 import decode_predictions

        res = decode_predictions(raw_prediction, top=3)
        str_template = "Best response of {best} with probability of {p}"
        return [str_template.format(best=i[0][1], p=i[0][2]) for i in res]

    def predict(self, context=None, model_input=None):
        """
        Wrapper predict method
        """
        n_records = model_input.shape[0]

        input_numpy = model_input.values
        raw_predictions = np.vectorize(self.predict_from_bytes, otypes=[np.ndarray])(input_numpy)
        raw_predictions = np.array(raw_predictions.tolist()).reshape([n_records, 1000])

        decoded_predictions = self.postprocess_raw_predictions(raw_predictions)
        decoded_predictions = pd.DataFrame(decoded_predictions, columns=["prediction"])
        decoded_predictions.index = model_input.index
        return decoded_predictions

classifier_pyfunc = KerasImageClassifierPyfunc()
classifier_pyfunc.load_context(path=model_uri) # This will run automatically when using mlflow.pyfunc.load_model()

output = classifier_pyfunc.predict(model_input=data)
output

# COMMAND ----------

# MAGIC %md Toilet tissue?  Take a look at the images to see why the model predicted these classes.  Note the confidence of the prediction.

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(filenames[0]) # The "toilet tissue"
plt.imshow(img)

# COMMAND ----------

img = mpimg.imread(filenames[1]) # The "tabby"
plt.imshow(img)

# COMMAND ----------

# MAGIC %md ## Save the **`pyfunc`** with Dependencies

# COMMAND ----------

# MAGIC %md Create a model signature to document model inputs and outputs.

# COMMAND ----------

type(output)

# COMMAND ----------

from mlflow.models.signature import infer_signature

signature = infer_signature(data, output)
signature

# COMMAND ----------

# MAGIC %md Create the Conda environment for all the **`pyfunc`**'s dependencies.

# COMMAND ----------

import cloudpickle
import tensorflow.keras
from sys import version_info

conda_env = {
    "channels": ["defaults"],
    "dependencies": [
      f"python={version_info.major}.{version_info.minor}.{version_info.micro}",
      "pip",
      {"pip": [
          "mlflow==1.30.0", # Using pre-2.0 for this notebook
          f"tensorflow=={tf.__version__}",
          f"cloudpickle==1.2.2", # Forcing cloudpickle version due to serialization issue
          f"keras=={tensorflow.keras.__version__}" # Need both tensorflow and keras due to mlflow dependency
        ],
      },
    ],
    "name": "keras_env"
}

conda_env

# COMMAND ----------

# MAGIC %md Create associated artifacts. Note that since the default serialization of a **`keras`** models uses **`tensorflow`** serialization we'll instead read in the model using **`keras`** when the Python function is loaded.

# COMMAND ----------

artifacts = {
    "keras_model": model_uri
}

# COMMAND ----------

# MAGIC %md Log the **`pyfunc`** including the artifacts, environment, signature, and input example.

# COMMAND ----------

with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
      "vgg-model", 
      python_model=KerasImageClassifierPyfunc(), 
      artifacts=artifacts,
      conda_env=conda_env,
      signature=signature,
      input_example=data[:1], # Can only log one row because of MLflow serving size limits
      registered_model_name=model_name # Registers model
  )

# COMMAND ----------

# MAGIC %md ## Load from the Model Registry and Serve using REST

# COMMAND ----------

# MAGIC %md Load from the model registry to confirm the registration is complete.

# COMMAND ----------

import time

model_version_uri = f"models:/{model_name}/1"

while True:
    try:
        model_version_1 = mlflow.pyfunc.load_model(model_version_uri)
        break
    except:
        print(f"Model not ready yet.  Sleeping...")
        time.sleep(10)
        
model_version_1.predict(data) # Test on sample data

# COMMAND ----------

# MAGIC %md ## Enable cluster serving.  
# MAGIC 
# MAGIC **This will create a dedicated VM to serve this model** so be sure to shut it down when you're done. 

# COMMAND ----------

# We need an endpoint at which to execute our requests which we can get from the notebook's API URL
api_url = mlflow.utils.databricks_utils.get_webapp_url()

# We need a token for the API, which we can get from the notebook as well
# As the best practice, we recommend using Databricks secret scopes for tokens
token = mlflow.utils.databricks_utils._get_command_context().apiToken().get()

# With the token, we can create our authorization header for our subsequent REST calls
headers = {"Authorization": f"Bearer {token}"}

# COMMAND ----------

import requests

url = f"{api_url}/api/2.0/mlflow/endpoints/enable"

r = requests.post(url, headers=headers, json={"registered_model_name": model_name})
assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

# COMMAND ----------

# MAGIC %md Get cluster serving status.  This will wait until the endpoint is ready.

# COMMAND ----------

import json

while True:
    url = f"{api_url}/api/2.0/mlflow/endpoints/get-status"
    r = requests.get(url, headers=headers, json={"registered_model_name": model_name})
    assert r.status_code == 200, f"Expected an HTTP 200 response, received {r.status_code}"

    state = r.json().get("endpoint_status")["state"]
    if state != "ENDPOINT_STATE_READY":
        print(f"Endpoint not ready yet ({state}).  Sleeping...")
        time.sleep(10)
    else:
        print(f"Endpoint READY")
        break

# Takes ~ 4min to start up
print(json.dumps(r.json(), indent=4))

# COMMAND ----------

# MAGIC %md
# MAGIC Score using the REST endpoint and wait until it's ready.
# MAGIC 
# MAGIC **Note:** this code could fail if the Conda environment is not fully running yet.  Retry if you receive a 404 error.

# COMMAND ----------

import os
import pandas as pd

def score_model(data: pd.DataFrame, model_name: str):
    url = f"{api_url}/model/{model_name}/1/invocations"
    data_json = data.to_dict(orient="split")
    r = requests.request(method="POST", headers=headers, url=url, json=data_json)
    assert r.status_code in [200, 404], f"Expected an HTTP 200 or 404 response, received {r.status_code}"
    return r.json() if r.status_code == 200 else None

results = score_model(data, model_name)

while not results:
    print(f"Conda environment still building.  Sleeping...")
    time.sleep(10)
    results = score_model(data, model_name)
    
print(json.dumps(results, indent=4))

# COMMAND ----------

# MAGIC %md Disable cluster serving.  **This will shut down the model serving endpoint.**

# COMMAND ----------

url = f"{api_url}/api/2.0/mlflow/endpoints/disable"
requests.post(url, headers=headers, json={"registered_model_name": model_name})

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Classroom Cleanup
# MAGIC 
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
