-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md # Higher Order Functions
-- MAGIC 
-- MAGIC Higher order functions in Spark SQL allow you to work directly with complex data types. When working with hierarchical data, records are frequently stored as array or map type objects. This lesson will demonstrate how to use higher-order functions to transform, filter, and flag array data while preserving the original structure.
-- MAGIC 
-- MAGIC ## Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC - Apply higher-order functions (`TRANSFORM`, `FILTER`, `EXISTS`) to transform data

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-load

-- COMMAND ----------

-- MAGIC %md
-- MAGIC These examples use data from the `sales` table of the ecommerce dataset.

-- COMMAND ----------

DESCRIBE sales

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Filter
-- MAGIC 
-- MAGIC `FILTER` filters an array using the given lambda function.
-- MAGIC 
-- MAGIC Let's say we want to remove items that are not king-sized from all records in our `items` column. We can use the `FILTER` function to create a new column that excludes that value from each array.
-- MAGIC 
-- MAGIC **`FILTER (items, i -> i.item_id LIKE "%K") AS king_items`**
-- MAGIC 
-- MAGIC In the statement above:
-- MAGIC - **`FILTER`** : the name of the higher-order function <br>
-- MAGIC - **`items`** : the name of our input array <br>
-- MAGIC - **`i`** : the name of the iterator variable. You choose this name and then use it in the lambda function. It iterates over the array, cycling each value into the function one at a time.<br>
-- MAGIC - **`->`** :  Indicates the start of a function <br>
-- MAGIC - **`i.item_id LIKE "%K"`** : This is the function. Each value is checked to see if it ends with the capital letter K. If it is, it gets filtered into the new column, `king_items`

-- COMMAND ----------

-- filter for sales of only king sized items
SELECT
  order_id,
  items,
  FILTER (items, i -> i.item_id LIKE "%K") AS king_items
FROM
  sales

-- COMMAND ----------

-- MAGIC %md
-- MAGIC You may write a filter that produces a lot of empty arrays in the created column. When that happens, it can be useful to use a `WHERE` clause to show only non-empty array values in the returned column. 
-- MAGIC 
-- MAGIC In this example, we accomplish that by using a subquery (a query within a query). They are useful for performing an operation in multiple steps. In this case, we're using it to create the named column that we will use with a `WHERE` clause. 

-- COMMAND ----------

-- filter for sales of only king sized items
SELECT
  order_id,
  king_items
FROM
  (
    SELECT
      order_id,
      FILTER (items, i -> i.item_id LIKE "%K") AS king_items
    FROM
      sales
  )
WHERE
  size(king_items) > 0 -- only include sale records with at least one king sized item

-- COMMAND ----------

-- Create temporary view for sales of king sized items
CREATE OR REPLACE TEMP VIEW king_size_sales AS

SELECT
  order_id,
  king_items
FROM
  (
    SELECT
      order_id,
      FILTER (items, i -> i.item_id LIKE "%K") AS king_items
    FROM
      sales
  )
WHERE
  size(king_items) > 0;
  
SELECT * FROM king_size_sales

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ### Exists
-- MAGIC `EXIST` tests whether a statement is true for one or more elements in an array.
-- MAGIC 
-- MAGIC Let's flag all sales with `"Mattress"` in the item_name field.
-- MAGIC 
-- MAGIC **`EXISTS (items, i -> i.item_name LIKE "%Mattress") AS mattress`**
-- MAGIC 
-- MAGIC In the statement above:
-- MAGIC - **`EXISTS`** : the name of the higher-order function <br>
-- MAGIC - **`items`** : the name of our input array <br>
-- MAGIC - **`i`** : the name of the iterator variable. You choose this name and then use it in the lambda function. It iterates over the array, cycling each value into the function one at a time.<br>
-- MAGIC - **`->`** :  Indicates the start of a function <br>
-- MAGIC - **`i.item_name LIKE "%Mattress"`** : This is the function. Each value is checked to see if the item_name ends with "Mattress." If it is, it gets flagged into the new column, `mattress`
-- MAGIC 
-- MAGIC Let's do the same for pillow items as well.

-- COMMAND ----------

-- extract column values from elements in array
SELECT
  items,
  EXISTS (items, i -> i.item_name LIKE "%Mattress") AS mattress,
  EXISTS (items, i -> i.item_name LIKE "%Pillow") AS pillow
FROM
  sales

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Transform
-- MAGIC 
-- MAGIC `TRANSFORM` uses the given lambda function to transform all elements in an array.
-- MAGIC 
-- MAGIC Built-in functions are designed to operate on a single, simple data type within a cell; they cannot process array values. `TRANSFORM` can be particularly useful when you want to apply an existing function to each element in an array. 
-- MAGIC 
-- MAGIC Let's compute the total revenue from king-sized items per order.
-- MAGIC 
-- MAGIC **`TRANSFORM(king_items, k -> CAST(k.item_revenue_in_usd * 100 AS INT)) AS item_revenues`**
-- MAGIC 
-- MAGIC In the statement above:
-- MAGIC - **`TRANSFORM`** : the name of the higher-order function <br>
-- MAGIC - **`king_items`** : the name of our input array <br>
-- MAGIC - **`k`** : the name of the iterator variable. We choose this name and then use it in the lambda function. It iterates over the array, cycling each value into the function one at a time. Note that we're using the same kind as references as in the previous command, but we name the iterator with a new variable<br>
-- MAGIC - **`->`** :  Indicates the start of a function <br>
-- MAGIC - **`CAST(k.item_revenue_in_usd * 100 AS INT)`** : This is the function. For each value in the input array, we extract the item's revenue value, multiply it by 100, and cast the result to integer.

-- COMMAND ----------

-- get total revenue from king items per order
SELECT
  order_id,
  king_items,
  TRANSFORM (
    king_items,
    k -> CAST(k.item_revenue_in_usd * 100 AS INT) -- extract revenue of each item in the king items list
  ) AS item_revenues
FROM
  king_size_sales

-- COMMAND ----------

-- MAGIC %md We'll save this as a temporary view for the next section.

-- COMMAND ----------

-- get total revenue from king items per order
CREATE OR REPLACE TEMP VIEW king_item_revenues AS

SELECT
  order_id,
  king_items,
  TRANSFORM (
    king_items,
    k -> CAST(k.item_revenue_in_usd * 100 AS INT)
  ) AS item_revenues
FROM
  king_size_sales;

SELECT * FROM king_item_revenues

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC ### Reduce 
-- MAGIC `REDUCE` is more advanced than `TRANSFORM`; it takes two lambda functions. We can use it to reduce the elements of an array to a single value by merging the elements into a buffer, and applying a finishing function on the final buffer. 
-- MAGIC 
-- MAGIC We will use the reduce function to find the total revenue from king sized items per order.
-- MAGIC 
-- MAGIC **`REDUCE(item_revenues, 0, (r, acc) -> r + acc, acc ->(round(acc / 100, 2))) AS total_king_revenue`**
-- MAGIC 
-- MAGIC In the statement above:
-- MAGIC - **`item_revenues`** is the input array<br>
-- MAGIC - **`0`** is the starting point for the buffer. Remember, we have to hold a temporary buffer value each time a new value is added to from the array; we start at zero in this case to get an accurate sum of the values in the list. <br>
-- MAGIC - **`(r, acc)`** is the list of arguments we'll use for this function. It may be helpful to think of `acc` as the buffer value and `r` as the value that gets added to the buffer.<br>
-- MAGIC - **`r + acc`** is the buffer function. As the function iterates over the list, it holds the total (`acc`) and adds the next value in the list (`r`). <br>
-- MAGIC - **`round(acc / 100, 2)`** is the finishing function. Once we have the sum of all numbers in the array, we divide by 100 and round to two decimal points to get the total revenue value in usd. <br>

-- COMMAND ----------

-- get total revenue from king items per order
CREATE OR REPLACE TEMP VIEW total_king_revenues AS

SELECT
  order_id,
  king_items,
  REDUCE (
    item_revenues,
    0,
    (r, acc) -> r + acc,        -- add revenue of each item
    acc -> round(acc / 100, 2)  -- divide total by 100 and round to two decimal points for value in usd
  ) AS total_king_revenue
FROM
  king_item_revenues;

SELECT * FROM total_king_revenues

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
