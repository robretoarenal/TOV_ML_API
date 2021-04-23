#!/bin/bash
#Running the docker-compose
echo "killing old docker processes"
docker-compose rm -fs
echo "building docker containers"
docker-compose up --build -d
