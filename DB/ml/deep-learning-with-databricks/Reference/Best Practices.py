# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Best Practices

# COMMAND ----------

# MAGIC %md
# MAGIC ## Neural Network Architecture
# MAGIC 
# MAGIC - Shape
# MAGIC   - Number of hidden layers
# MAGIC     - Inverted pyramid shape: start out wide and narrow down later
# MAGIC     - Hourglass shape: start out wide, narrow down in the the middle layers, and widen towards the end (common encoder-decoder structure)
# MAGIC     <br>
# MAGIC     <img src="http://files.training.databricks.com/images/hourglass_architecture.png" width="500" height="500">
# MAGIC   - Number of units/neurons in the input and output layers
# MAGIC     - Depends on the input and output your task requires. Example: the MNIST task requires 28 x 28 = 784 input units and 10 output units 
# MAGIC   - Better to increase the number of layers instead of the number of neurons/units per layer
# MAGIC   - Play with the size systematically to compare the performances -- it's a trial-and-error process
# MAGIC   - Two typical approaches:
# MAGIC     - Increase the number of neurons and/or layers until the network starts overfitting
# MAGIC     - Use more layers and neurons than you need, then use early stopping and other regularization techniques to prevent overfitting
# MAGIC   - Borrow ideas from research papers
# MAGIC - Learning Rates:
# MAGIC   - Slanted triangular learning rates: Linearly increase learning rate, followed by linear decrease in learning rate
# MAGIC   <br>
# MAGIC   <img src="https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/slanted_triangular_lr+.png" height="200" width="400">
# MAGIC   - Discriminative Fine Tuning: Different LR per layer
# MAGIC   - Learning rate warmup (start with low LR and change to a higher LR)
# MAGIC   - When the batch size is larger, scale up the learning rate accordingly
# MAGIC   - Use a learning rate finder or scheduler (examples are <a href="https://www.avanwyk.com/finding-a-learning-rate-in-tensorflow-2/" target="_blank">here</a> and <a href="http://d2l.ai/chapter_optimization/lr-scheduler.html#schedulers" target="_blank">here</a> 
# MAGIC - Batch normalization works best when done:
# MAGIC   - after activation functions
# MAGIC   - after dropout layers (if dropout layers are used concurrently)
# MAGIC   - after convolutional or fully connected layers
# MAGIC   
# MAGIC - Neural Network Architecture Search
# MAGIC   - <a href="https://arxiv.org/abs/1611.01578" target="_blank">Neural Architecture Search with Reinforcement Learning</a>
# MAGIC   - It is very expensive to do neural architecture search! **Do as much work as you can on a small data sample**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regularization Techniques <br>
# MAGIC 
# MAGIC - Dropout
# MAGIC   - Apply to most type of layers (e.g. fully connected, convolutional, recurrent) and larger networks
# MAGIC   - Set an additional probability for dropping each node (typically set between 0.1 and 0.5)
# MAGIC     - .5 is a good starting place for hidden layers
# MAGIC     - .1 is a good starting place for the input layer
# MAGIC   - Increase the dropout rate for large layers and reduce it for smaller layers
# MAGIC - Early stopping
# MAGIC <br>
# MAGIC  <img src="https://miro.medium.com/max/1247/1*2BvEinjHM4SXt2ge0MOi4w.png" width="500" height="300">
# MAGIC - Reduce learning rate over time
# MAGIC - Weight decay
# MAGIC - Data augmentation
# MAGIC - L1 or L2
# MAGIC 
# MAGIC Note: No regularization on bias and no weight decay for bias terms <br>
# MAGIC 
# MAGIC Click to read the <a href="https://www.tensorflow.org/tutorials/keras/overfit_and_underfit#training_procedure" target="_blank">Tensorflow documentation</a> for code examples.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convolutional Neural Network
# MAGIC 
# MAGIC - Layers are typically arranged so that they gradually decrease the spatial resolution of the representations, while increasing the number of channels
# MAGIC - Don't use random initialization. Use <a href="https://arxiv.org/pdf/1502.01852.pdf" target="_blank">He Initalization</a> to keep the variance of activations constant across every layer, which helps prevent exploding or vanishing gradients
# MAGIC - Label Smoothing: introduces noise for the labels.
# MAGIC <br>
# MAGIC <img src="https://paperswithcode.com/media/methods/image3_1_oTiwmLN.png">
# MAGIC - Max pooling generally performs better than average pooling 
# MAGIC - Image Augmentation:
# MAGIC   - Random crops of rectangular areas in images
# MAGIC   - Random flips
# MAGIC   - Adjust hue, saturation, brightness
# MAGIC   - Note: Use the right kind of augmentation (e.g. don't flip a cat upside down, but satellite image OK)
# MAGIC   
# MAGIC   
# MAGIC ### Bag of Tricks (<a href="https://arxiv.org/pdf/1812.01187.pdf" target="_blank">Paper</a>)
# MAGIC 
# MAGIC - Efficient training
# MAGIC   - Large batch training with linear scaling learning rate, LR warmup, zero γ, no bias decay
# MAGIC   - Low precision training
# MAGIC   - Model tweaks
# MAGIC - Switching stride size of first two conv. Layer
# MAGIC   - Replacing 7x7 conv with three 3x3 convs 
# MAGIC   - Adding 2x2 average pooling layer with stride of 2 before conv 
# MAGIC - Training refinements 
# MAGIC   - Cosine LR decay
# MAGIC   - Label smoothing
# MAGIC   - Knowledge distillation
# MAGIC   - Mixup training
# MAGIC - Transfer learning
# MAGIC   - Applications in object detection and semantic segmentation

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transfer Learning
# MAGIC 
# MAGIC - Gradual Unfreezing: Unfreeze last layer and train for one epoch and keep unfreezing layers until all layers trained/terminal layer
# MAGIC - Specifically for image use cases:
# MAGIC   - Use a differential learning rate, where the learning rate is determined on a per-layer basis. 
# MAGIC     - Assign lower learning rates to the bottom layers responding to edges and geometrics
# MAGIC     - Assign higher learning rates to the layers responding to more complex features

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scaling Deep Learning Best Practices
# MAGIC 
# MAGIC * Use a GPU
# MAGIC * Use Petastorm
# MAGIC * Use Multiple GPUs with Horovod
# MAGIC 
# MAGIC Click <a href="https://databricks.com/blog/2019/08/15/how-not-to-scale-deep-learning-in-6-easy-steps.html" target="_blank">here</a> to read the Databricks blog post.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Optimizing input pipeline performance with the <a href="https://www.tensorflow.org/api_docs/python/tf/data" target="_blank">tf.data</a> API
# MAGIC 
# MAGIC The <a href="https://www.tensorflow.org/api_docs/python/tf/data" target="_blank">**`tf.data`**</a> API facilitates the building of flexible and efficient input pipelines during model training. More often than not, how optimally batches are fed into your GPU during training will be the core determinant of how efficiently you utilize the underlying GPU resources you are training on. As such, optimizing the input pipeline to your model will be crucial when it comes to achieving peak performance and optimal utilization of your compute resources.
# MAGIC 
# MAGIC The **`tf.data`** API is the recommended API to use when creating input pipelines for TensorFlow models. It contains a <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset" target="_blank">**`tf.data.Dataset`**</a> abstraction that represents a sequence of elements, which are fed into your training process. If you are unfamiliar with the **`tf.data`** API, the <a href="https://www.tensorflow.org/guide/data" target="_blank">following guide</a> is a great resource for getting started on building these input pipelines.
# MAGIC 
# MAGIC With the **`tf.data`** API you can avail of a range of functionalities to optimize the performance of these pipelines. <a href="https://www.tensorflow.org/guide/data_performance" target="_blank">This guide</a> is an excellent primer on how to improve the performance of **`tf.data`** pipelines.
# MAGIC 
# MAGIC Note that Petastorm - via its <a href="https://petastorm.readthedocs.io/en/latest/api.html?highlight=make%20tf%20dataset#petastorm.spark.spark_dataset_converter.SparkDatasetConverter" target="_blank">**`spark_dataset_converter`**</a> - will create a **`tf.data.Dataset`** when using <a href="https://petastorm.readthedocs.io/en/latest/api.html?highlight=make%20tf%20dataset#petastorm.spark.spark_dataset_converter.SparkDatasetConverter.make_tf_dataset" target="_blank">**`make_tf_dataset`**</a>, and under the hood will handle the caching, buffering and batching of your dataset which you would ordinarily configure with the **`tf.data`** API.
# MAGIC 
# MAGIC ##### *How do I monitor the utilization of my GPU?*
# MAGIC 
# MAGIC Utilization metrics can be monitored during model training via the <a href="https://docs.databricks.com/clusters/clusters-manage.html#ganglia-metrics-1" target="_blank">Ganglia UI</a>. Optimal performance of your input pipeline will result in minimal pauses during your training procedure.

# COMMAND ----------

# MAGIC %md
# MAGIC #### References
# MAGIC 
# MAGIC - <a href="https://arxiv.org/pdf/1801.06146.pdf" target="_blank">ULMFiT - Language Model Fine-tuning</a>
# MAGIC - <a href="https://arxiv.org/pdf/1812.01187.pdf" target="_blank">Bag of Tricks for CNN</a>
# MAGIC - <a href="https://forums.fast.ai/t/30-best-practices/12344" target="_blank">fast.ai</a>
# MAGIC - <a href="https://medium.com/starschema-blog/transfer-learning-the-dos-and-donts-165729d66625" target="_blank">Transfer Learning: The Dos and Don'ts</a>
# MAGIC - <a href="https://link.springer.com/article/10.1007/s11042-019-08453-9" target="_blank">Dropout vs Batch Normalization</a>
# MAGIC - <a href="https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/" target="_blank">How to Configure the Number of Layers and Nodes</a>
# MAGIC - <a href="https://www.oreilly.com/library/view/neural-networks-and/9781492037354/ch01.html" target="_blank">Neural networks and Deep Learning by Aurélien Géron</a>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
