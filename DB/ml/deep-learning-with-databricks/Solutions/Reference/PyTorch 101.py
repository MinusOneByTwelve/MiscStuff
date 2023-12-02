# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # PyTorch 101
# MAGIC 
# MAGIC In this lesson you explore the basics of PyTorch and train a simple neural network to predict the quality of wine.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Create PyTorch tensors
# MAGIC  - Explore gradients and DAGs 
# MAGIC  - Load data in a PyTorch custom class
# MAGIC  - Create a PyTorch model
# MAGIC  - Train and evaluate the model

# COMMAND ----------

# MAGIC %md ### Tensors
# MAGIC 
# MAGIC Before we create a PyTorch model, look at a PyTorch tensor. Tensors are a core datatype in PyTorch. Any data used with a PyTorch model and all of the model weights must be of this type. 
# MAGIC 
# MAGIC We can cast other similar data types such as numpy arrays and python lists to PyTorch tensors. 

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md Create a tensor from a list.

# COMMAND ----------

import torch 
import numpy as np

example_data = [1, 2, 3], [4, 5, 6], [7, 8, 9]
example_tensor = torch.tensor(example_data)
example_tensor

# COMMAND ----------

# MAGIC %md Create a tensor from a numpy array.

# COMMAND ----------

example_numpy_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
example_tensor_from_numpy = torch.from_numpy(example_numpy_data)
example_tensor_from_numpy

# COMMAND ----------

# MAGIC %md Check the shape and type of the tensors.

# COMMAND ----------

print(example_tensor.shape)
print(example_tensor.dtype)

# COMMAND ----------

# MAGIC %md ## Gradients in PyTorch
# MAGIC 
# MAGIC In deep learning it is crucial to be able to calculate the gradient of the loss function for backpropagation during training. In order to provide this functionality, PyTorch's **`torch.autograd`** dynamically constructs a DAG of tensors and functions. 
# MAGIC 
# MAGIC In order to become a part of the DAG, a tensor must have the **`required_grad`** parameter set to **`True`**. It is **`False`** by default. 
# MAGIC 
# MAGIC Construct a tensor we want to add to the DAG. 
# MAGIC 
# MAGIC **Note:** The tensor must be of dtype float. 

# COMMAND ----------

tensor_x = torch.tensor([1.], requires_grad=True)
tensor_x

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Right now our DAG looks like this: 
# MAGIC 
# MAGIC  ![Tensor 1](http://files.training.databricks.com/images/mlpupdates/tensor1.png)
# MAGIC 
# MAGIC Any operation involving the tensor, the function of the operation, and the output will be tracked in the DAG. This includes weights, biases, gradients, and activation functions.
# MAGIC 
# MAGIC Here we define **`tensor_y`** to be **`2 * tensor_x`**.

# COMMAND ----------

tensor_y = 2 * tensor_x

tensor_y

# COMMAND ----------

# MAGIC %md Now the DAG looks like this:
# MAGIC   
# MAGIC ![Tensor 1](http://files.training.databricks.com/images/mlpupdates/tensor2b.png)
# MAGIC 
# MAGIC We have now defined **`y = 2x`**. The derivative of y with respect to x here is 2. However, if we check **`tensor_x.grad`** we don't see anything yet. 

# COMMAND ----------

print(tensor_x.grad)

# COMMAND ----------

# MAGIC %md Tensors have a **`.backward()`** method that populates this gradient field. 
# MAGIC 
# MAGIC When the tensor at the end of this computational graph, **`tensor_y`** in our case, calls **`.backward()`**, **`torch.autograd()`** will go backwards using the dynamic DAG and chain rule to populate the gradient fields.
# MAGIC 
# MAGIC For our example, once we call **`tensor_y.backward()`**, **`torch.autograd`** will automatically handle the backpropagation and populate **`tensor_x.grad`**
# MAGIC 
# MAGIC **Note:** By default it only populates the gradient field of the leaf nodes (initial inputs) to the DAG. 

# COMMAND ----------

tensor_y.backward()

print(tensor_x.grad)

# COMMAND ----------

# MAGIC %md We can see here that **`tensor_x.grad`** is now 2 as expected. 

# COMMAND ----------

# MAGIC %md ## Dataset Loading
# MAGIC 
# MAGIC Now that we understand basic data structures in PyTorch, we'll go through an example of training a regression model on the Wine Quality dataset. 
# MAGIC 
# MAGIC PyTorch expects datasets to be of its custom **`Dataset`** class. In order to load in our data, we will have to inherit from this class and override the following methods:<br><br>
# MAGIC 
# MAGIC 1. **`__init__`**: a typical Python **`__init__`** run when instantiating an object of our class
# MAGIC 2. **`__len__`**: returns the length of our dataset.
# MAGIC 3. **`__getitem__`**: returns the features and label of the dataset at a given index **`idx`**. In PyTorch you do not separate the features and labels into an X, y split like with packages such as **`sklearn`**. Instead you will split the dataset later into train and test but this function is used to index into the dataset and return the X, y pair. The end of this method should return **`features, label`** for the dataset at the given index. 
# MAGIC 
# MAGIC We will also include a **`get_splits`** helper function. PyTorch provides a **`random_splits`** function to do your train test split, but it is done by passing in raw counts you want in each side of the split instead of the fraction in each. This helper will take in the fraction in the test set, calculate the counts based on the size of the dataset, and then call **`random_split`** with those counts. 

# COMMAND ----------

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import random_split

class MyDataset(Dataset):
  
    def __init__(self, data):
        y = pd.DataFrame(data.target)
        X = pd.DataFrame(data.data)
        X = X.apply(lambda x: (x - x.mean()) / x.std()) # Standardize the dataset
        self.X = torch.from_numpy(X.values.astype("float32"))
        self.y = torch.from_numpy(pd.DataFrame(y).values.astype("float32"))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self, n_test=0.2):
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

# COMMAND ----------

# MAGIC %md For more details:<br><br>
# MAGIC 
# MAGIC * **`__init__`**: Load in our dataset from the path, drop the index column, separate the features and label, and standardize the features. Then save a class attribute X and y with the features and label respectively. 
# MAGIC * **`__len__`**: Return the length of the dataset by returning the number of rows in the dataframe.
# MAGIC * **`__getitem__`**: Take in index **`idx`** and return the row of that number in the features dataframe X and the row in the features dataframe y.
# MAGIC * **`get_splits`**: Take in the fraction of observations we want in our test set **`n_test`**, calculate the number of rows that correspond to that fraction given our dataset length, and return **`random_split`** on the dataset with those raw numbers. 
# MAGIC 
# MAGIC More information can be found <a href="https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files" target="_blank">here</a>.

# COMMAND ----------

from sklearn.datasets import fetch_california_housing

cal_housing = fetch_california_housing()
dataset = MyDataset(cal_housing)
train_data, test_data = dataset.get_splits()

# COMMAND ----------

# MAGIC %md Now we have our dataset train/test splits. However, we will want to do a batch model training. In order to do this, we need to create a PyTorch **`Dataloader`** object with our training set. 
# MAGIC 
# MAGIC Pass in the batch size and a shuffle parameter to indicate whether we want the batches to be shuffled across epochs.

# COMMAND ----------

from torch.utils.data import DataLoader

train_dl = DataLoader(train_data, batch_size=64, shuffle=True)

# COMMAND ----------

# MAGIC %md ## Create a PyTorch Model
# MAGIC 
# MAGIC There are multiple ways to create PyTorch neural networks. 
# MAGIC 
# MAGIC For simple neural networks of sequential layers without custom changes, the easiest way is to use **`nn.Sequential`**. We will use this method. 
# MAGIC 
# MAGIC If you wanted to use custom logic, you would inherit from **`nn.Module`** and override the **`forward`** method. More information can be found <a href="https://pytorch.org/docs/stable/generated/torch.nn.Module.html" target="_blank">here</a>.
# MAGIC 
# MAGIC Our model will simply be a dense network with one hidden layer. 

# COMMAND ----------

import torch.nn as nn

model = nn.Sequential(nn.Linear(8, 10), nn.ReLU(), nn.Linear(10, 1))

print(model)

# COMMAND ----------

# MAGIC %md We also initialize our model weights with the xavier uniform distribution to prevent gradient explosion and/or vanishing.

# COMMAND ----------

torch.nn.init.xavier_uniform_(model[0].weight)

# COMMAND ----------

torch.nn.init.xavier_uniform_(model[2].weight)

# COMMAND ----------

# MAGIC %md ## Training and Inference
# MAGIC 
# MAGIC Now train our model. In PyTorch, there is no **`.fit()`** method to train your model. 
# MAGIC 
# MAGIC PyTorch is based around the dynamic DAG we described above, so training a neural network involves a lower level implementation where you manually write the training loop. We will put it in a helper function to simplify the code. These are the steps in this training function:<br><br>
# MAGIC 
# MAGIC 1. Define the loss function (MSE) and optimizer (ADAM)
# MAGIC 2. Enumerate over the dataloader object to loop over the batches. For each batch:
# MAGIC   1. Get the output of the model on the batch and calculate the loss.
# MAGIC   2. **`optimizer.zero_grad()`** to clear the gradients learned from the previous training step.
# MAGIC   3. **`loss.backward()`** to go back through the DAG and populate the gradients of the tensors.
# MAGIC   4. **`optimizer.step()`** to take one training step where weights are updated using the gradients populated with **`loss.backward()`**. 
# MAGIC   5. Print the loss every few batches to track training progress. 

# COMMAND ----------

# Step 1: Define the loss function (MSE) and optimizer (ADAM)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=[0.9, 0.999], eps=1e-8)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # switch to training mode

    # Step 2: Enumerate over the dataloader object to loop over the batches
    for batch, (X, y) in enumerate(dataloader):

        # Step 3.1: Get the output of the model on the batch and calculate the loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Step 3.2: clears the gradients learned from the previous training step
        optimizer.zero_grad() 

        # Step 3.3: go backwards through the DAG to populate the gradients of the tensors
        loss.backward()

        # Step 3.4: take one training step where we update the weights using the gradients
        optimizer.step()  

    loss = loss.item()
    print(f"Train loss: {round(loss, 2)}")

# COMMAND ----------

# MAGIC %md Define our test function. This loops over each batch in the test **`Dataloader`** and average the loss over the batches. 
# MAGIC 
# MAGIC **Note:** When performing inference, it is a best practice to call **`model.eval()`** and **`torch.no_grad()`**. Some layers and parts of the model behave differently during inference vs. training. Calling **`model.eval()`** sets these correctly. **`torch.no_grad()`** turns off gradient computation which isn't needed during inference.

# COMMAND ----------

def test(data, model, loss_fn):
    model.eval() # switch to inference mode
    with torch.no_grad(): # Don't need gradients for inference
        pred = model(data.dataset.X)
        loss = loss_fn(pred, data.dataset.y)

    print(f"Test loss: {round(loss.item(), 2)} \n")

# COMMAND ----------

# MAGIC %md Now train the model and evaluate the results.

# COMMAND ----------

epochs = 20

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dl, model, loss_fn, optimizer)
    test(test_data, model, loss_fn)
    
print("Done!")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
