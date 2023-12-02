source ~/.zshrc
bdc build.yaml -o
cp ~/tmp/curriculum/mlflow-exp-tracking-1.0.0/StudentFiles/Labs.dbc .
databricks workspace delete /Users/$USER@databricks.com/mlflow-exp-tracking-1.0.0 -r
databricks workspace import Labs.dbc /Users/$USER@databricks.com/mlflow-exp-tracking-1.0.0 -f DBC -l PYTHON
rm Labs.dbc
