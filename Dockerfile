FROM tensorflow/tensorflow:latest-py3

## Install python packages.
WORKDIR /tmp
COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r ./requirements.txt
RUN rm -r $HOME/.cache/pip

## Insall Jupyter
RUN pip3 install jupyter

## Set working directory to the head of project.
WORKDIR $HOME/kardionet/
