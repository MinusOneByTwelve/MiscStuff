# Spark Local Testing

This project demonstrates a method for using Docker and Docker Compose to do local unit 
and integration testing of Spark code. 

## Docker Image

A docker image for running Spark locally is defined by the `Dockerfile` in this repo in the directory `docker`.

This image is also available via Docker Hub: 
https://hub.docker.com/repository/docker/databricksacademy/spark-local-testing

## Run Tests using Docker

Pull or build the image:
1. **pull** `docker pull databricksacademy/spark-local-testing`
2. **build** `docker build -t docker databricksacademy/spark-local-testing`

Run `pytest`.

```
docker run -it -e STAGE=local \
  -v $(pwd):/home/databricks/work \
  databricksacademy/spark-local-testing pytest -s
```

Note that the sample source code uses an environment flag for `STAGE`.

## Run Tests using Docker-Compose

Include the `docker` directory and the `docker-compose.yml` file in your project. 

Then run the following to run a single test file:

```
docker-compose run spark_testing pytest -s <TESTFILE>
```

and this to run them all:

```
docker-compose run spark_testing pytest -s
```

## `Makefile`

A `Makefile` is included containing these commands.

## Debugging

Add a debugging breakpoint in a test or source code file with the command:

```
import ipdb; ipdb.set_trace()
```

This will launch the [IPython Debugger](https://github.com/gotcha/ipdb).
