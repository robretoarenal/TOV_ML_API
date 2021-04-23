#Create a base image.
FROM python:3.8 AS base

#Set work directory
WORKDIR /usr/src/flask_app

#Install PIP first
#RUN set -xe \
#    && apt-get update \
#    && apt-get install python-pip

#Update pip
RUN pip install --upgrade pip

#Install dependencies using a requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
