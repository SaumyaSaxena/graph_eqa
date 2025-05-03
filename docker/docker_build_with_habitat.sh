#!/bin/bash
#
# Script to build the docker image
# -u        Upload the build image to hub.docker.com

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source "${DIR}/docker_tag.sh"

docker_registry_image=grapheqa/grapheqa_for_habitat

docker build \
       --build-arg TAG=${TAG} \
       --build-arg BUILD_WITH_CUDA=OFF \
       --platform linux/amd64 \
       -t ${docker_registry_image}:${TAG} -f Dockerfile_habitat .

# Explicitly docker logout and docker login to make sure the correct docker repo will be pushed to if upload is needed