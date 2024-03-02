#!/bin/bash

# Usage: ./push_docker_images.sh -v <version>

while getopts v: flag
do
    case "${flag}" in
        v) version=${OPTARG};;
    esac
done

if [ -z "$version" ]; then
    echo "Version not specified. Use -v option to specify the version."
    exit 1
fi

# Define your Docker Hub username or organization
DOCKER_HUB_USERNAME="arbidev"

# Define service names and their Dockerfile locations
declare -A services=(
    ["arbi-tr-api-service"]="."
    ["arbi-tr-frontend"]="./frontend"
)

# Loop through the services to build, tag, and push images
for service in "${!services[@]}"
do
    dockerfile_location=${services[$service]}
    image_name="${DOCKER_HUB_USERNAME}/${service}"

    echo "Building ${service} from ${dockerfile_location}"
    docker build -t ${image_name}:${version} -t ${image_name}:latest ${dockerfile_location}

    echo "Pushing ${image_name}:${version}"
    docker push ${image_name}:${version}

    echo "Pushing ${image_name}:latest"
    docker push ${image_name}:latest
done

echo "All images have been pushed to Docker Hub."
