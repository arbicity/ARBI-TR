#!/bin/bash

docker build -t arbi-tr:v1 .
docker run -p 8000:8000 --gpus all arbi-tr:v1
