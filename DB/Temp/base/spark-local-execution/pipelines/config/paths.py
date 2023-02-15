import os

bases = {
    "local": "/home/databricks/work/tests/data/",
    "development": "/mnt/dbacademy/",
    "production": None,
}
env = os.environ["STAGE"]
base = bases[env]

test_raw = base + "test/raw"
test_bronze = base + "test/bronze"
test_silver = base + "test/silver"
test_gold = base + "test/gold/"
bronze = base + "bronze"
silver = base + "silver"
gold = base + "gold/"
checkpoints = base + "checkpoints/"
bronze_checkpoint = checkpoints + "bronze"
silver_checkpoint = checkpoints + "silver"
test_bronze_checkpoint = checkpoints + "test_bronze"
test_silver_checkpoint = checkpoints + "test_silver"
gold_checkpoint = checkpoints + "gold/"
