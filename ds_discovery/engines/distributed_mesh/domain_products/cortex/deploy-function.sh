#!/usr/bin/env bash -eu
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_IMAGE=controller-skill # Docker Image name


TAG="$(cat /dev/urandom | env LC_CTYPE=C tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1)"
CURRENT_PROFILE=$(cat ${HOME}/.cortex/config | jq -r ".currentProfile")
CORTEX_TOKEN=$(cat ${HOME}/.cortex/config | jq ".profiles" | jq ".[\"${CURRENT_PROFILE}\"]" | jq -r '.token')
REGISTRY=private-registry.dci-dev.dev-eks.insights.ai
REGISTRY_PASS=${CORTEX_TOKEN}
REGISTRY_USER=docker
docker login -u ${REGISTRY_USER} -p ${REGISTRY_PASS} https://${REGISTRY}
docker build -t ${DOCKER_IMAGE} .
IMAGENAME=${REGISTRY}/${DOCKER_IMAGE}:${TAG}
docker tag ${DOCKER_IMAGE} ${IMAGENAME}
docker push ${IMAGENAME}


cortex actions deploy --actionName ${DOCKER_IMAGE}  --actionType job --docker ${IMAGENAME}
