FROM azul/zulu-openjdk:8

LABEL maintainer=@joshuacook,@databricks

ARG hadoop_version="3.2"
ARG password="localtesting"
ARG py4j_version="0.10.9"
ARG python_version="3.7"
ARG spark_checksum="E90B31E58F6D95A42900BA4D288261D71F6C19FA39C1CB71862B792D1B5564941A320227F6AB0E09D946F16B8C1969ED2DEA2A369EC8F9D2D7099189234DE1BE"
ARG spark_version="3.1.1"

ENV USER "databricks"

# Install Python
RUN apt-get clean && \
    apt-get update && \
    apt-get install -y \
        python${python_version} python${python_version}-dev python${python_version}-distutils \
        software-properties-common curl wget git && \
    update-alternatives --install /usr/bin/python python /usr/bin/python${python_version} 1 && \
    update-alternatives --set python /usr/bin/python${python_version} && \
    curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py
    
# make user
RUN useradd -ms /bin/bash $USER && usermod -aG sudo $USER
ENV MAIN_USER $USER

# add jupyter, set password
RUN mkdir /etc/jupyter /home/$USER/.jupyter && chown -R $USER:$USER /home/$USER/.jupyter
ADD https://raw.githubusercontent.com/jupyter/docker-stacks/master/base-notebook/jupyter_notebook_config.py /etc/jupyter/
RUN echo "c.NotebookApp.token = u'${password}'" > /home/$USER/.jupyter/jupyter_notebook_config.py && \
    chown $USER:$USER /home/$USER/.jupyter/jupyter_notebook_config.py /etc/jupyter/jupyter_notebook_config.py
ENV PATH "/home/${USER}/.local/bin:${PATH}"

# Spark installation
ENV APACHE_SPARK_VERSION="${spark_version}"
ENV HADOOP_VERSION="${hadoop_version}"
    
WORKDIR /tmp
RUN wget -q $(wget -qO- https://www.apache.org/dyn/closer.lua/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz\?as_json | \
    python -c "import sys, json; content=json.load(sys.stdin); print(content['preferred']+content['path_info'])") && \
    echo "${spark_checksum} *spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" | sha512sum -c - && \
    tar xzf "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz" -C /usr/local --owner root --group root --no-same-owner && \
    rm "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"

WORKDIR /usr/local
RUN ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" spark

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH="${SPARK_HOME}/python:${SPARK_HOME}/python/lib/py4j-${py4j_version}-src.zip" \
    SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH=$PATH:$SPARK_HOME/bin
    

# install jupyter, jupyterlab, and extensions
USER $USER
RUN mkdir -p /home/$USER/.jupyter/lab
RUN pip install --user \
        notebook==6.2.0 \
        jupyterlab==3.0.7

# give user permission to jupyter directory
USER root
RUN chown -R $USER:$USER /home/$USER/.jupyter

# Configure For Work
USER $USER
RUN mkdir /home/$USER/work
WORKDIR /home/$USER/work

# pip install
COPY requirements.txt .
RUN pip install --user -r requirements.txt
