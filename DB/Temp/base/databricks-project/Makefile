clean-pyc:
		find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf

launch-test-server:
		docker-compose up -d

run-test: clean-pyc
		docker-compose exec spark_testing pytest -s $(testfile)

run-tests: clean-pyc
		docker-compose exec spark_testing pytest -s
