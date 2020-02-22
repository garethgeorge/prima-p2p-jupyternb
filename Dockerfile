FROM jupyter/base-notebook

USER root 
RUN apt-get update && apt-get install git -y && apt-get install build-essential -y