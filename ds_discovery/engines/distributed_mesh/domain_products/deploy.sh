#!/usr/bin/env bash

set -eux;

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
IMAGE=projecthadron/cortex_skill
DOCKER_IMAGE_TAG=0.1.1

PROFILE=$(jq -r .currentProfile ~/.cortex/config)
PROJECT=$(jq -r .profiles.${PROFILE}.project ~/.cortex/config)
URL=$(jq -r .profiles.${PROFILE}.url ~/.cortex/config)

# TODO fix docker registry and action deployment once v6 supports this. Also skill and action deployment will be one command.
#DOCKERREG=`curl -s -H "Authorization: Bearer ${TOKEN}" ${URL}/v3/actions/_config | jq -r .config.dockerPrivateRegistryUrl`

DOCKER_IMAGE=${IMAGE}:${DOCKER_IMAGE_TAG}
cd $SCRIPT_DIR
docker build -t ${IMAGE}:${DOCKER_IMAGE_TAG}  -f ${SCRIPT_DIR}/Dockerfile .
#cortex docker login
docker push ${DOCKER_IMAGE}
cortex actions deploy --actionName ${IMAGE} --actionType job --docker ${DOCKER_IMAGE}
#sleep 10
#cortex actions describe ${IMAGE}