# Databricks Project Template

This project template is designed to facilitate the development, testing, and deployment of Apache Spark Data Engineering pipelines across environments from local development using your preferred IDE to deployment on your Databricks cluster.

## Project Structure

This project has the following structure to a depth of 2.

```
.
├── Makefile
├── README.md
├── docker-compose.yml
├── env
│   └── docker
├── example
├── scripts
│   └── development.py
├── src
│   ├── config.py
│   ├── operations.py
│   └── utility.py
└── tests
    ├── data
    └── spark
```

- **`Makefile`** - defines common commands to be executed on the repo, including launching a local development server and running tests.
- **`doc`** - contains documentation associated with this project
- **`docker-compose.yml`** - defines the local development docker services
- **`env/docker`** - contains the `Dockerfile` and `requirements.txt` used to define the Python environment for local development
- **`example`** - contains a built-out example for how to use this project structure
- **`scripts`** - contains python scripts used for exploration and development purposes (**TODO**) discuss how to use these with Databricks and with JupyterLab
- **`src`** - contains source code
- **`tests/data`** - contains fixture data used during testing
- **`tests/spark`** - contains unit and integration tests


## Development

### Launch Local Development Server

Local development is facilitated by Docker and Docker Compose and built as an extension to the `jupyter/pyspark-notebook` docker image.

To begin developing, start the development server using the following command:

```
make launch-test-server
```

This will launch a local single-node spark cluster. The password is `"local spark cluster"`.

This cluster can be interacted with using Jupyter Labs at [localhost:10000](localhost:10000).

The cluster is used for running local tests against the pyspark package being developed.

### Running Tests Locally

Run tests against the local package using the `make` commands below.

Run a single test file:

```
make run-test testfile=<PATH_TO_TEST_FILE>
```

Run all tests:

```
make run-tests
```
