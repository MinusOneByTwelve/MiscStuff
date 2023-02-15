clean-pyc:
		find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

docker-run-tests: clean-pyc
		docker build -t spark-local-testing docker 
		docker run -e STAGE=local -v ${CURDIR}:/home/databricks/work -it spark-local-testing pytest -s

run-test: clean-pyc
		docker-compose run spark_testing pytest -s $(testfile)

run-tests: clean-pyc
		docker-compose run spark_testing pytest -s
