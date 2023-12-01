// Databricks notebook source
// MAGIC %md ###Mount Data Lake Gen 2

// COMMAND ----------

val configs = Map(
  "fs.azure.account.auth.type" -> "OAuth",
  "fs.azure.account.oauth.provider.type" -> "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id" -> "<application-id>",
  "fs.azure.account.oauth2.client.secret" -> "<key-name-for-service-credential>"),
  "fs.azure.account.oauth2.client.endpoint" -> "https://login.microsoftonline.com/<directory-id>/oauth2/token")

// Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://<file-system-name>@<storage-account-name>.dfs.core.windows.net/",
  mountPoint = "/mnt/<mount-name>",
  extraConfigs = configs)

// COMMAND ----------

val configs = Map(
  "fs.azure.account.auth.type" -> "OAuth",
  "fs.azure.account.oauth.provider.type" -> "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
  "fs.azure.account.oauth2.client.id" -> "99474d89-563d-43ca-a65c-b642e9de03ec",
  "fs.azure.account.oauth2.client.secret" -> "Kh7M4.rFWuGAPwv[8UHYax4FAHgptv.:",
  "fs.azure.account.oauth2.client.endpoint" -> "https://login.microsoftonline.com/73ac2b77-a66b-432d-a9fc-02b175d014c1/oauth2/token")

// Optionally, you can add <directory-name> to the source URI of your mount point.
dbutils.fs.mount(
  source = "abfss://inputdata@gen2storage1088.dfs.core.windows.net/",
  mountPoint = "/mnt/DatalakeGen2",
  extraConfigs = configs)

// COMMAND ----------

//dbutils.fs.unmount("/mnt/DatalakeGen2")

// COMMAND ----------

display(
  dbutils.fs.ls("mnt/DatalakeGen2/")
)
