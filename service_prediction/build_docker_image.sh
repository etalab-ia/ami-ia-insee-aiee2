#!/usr/bin/env bash

########################
#  Build docker
#######################
set -e

mkdir -p src/data_import
cp -r ../data_import/*.py  src/data_import

mkdir -p src/nomenclatures/training_classes
cp -r ../nomenclatures/training_classes/*.py  src/nomenclatures/training_classes
cp -r ../nomenclatures/*.py  src/nomenclatures
grep -rl '^from train' src/nomenclatures | xargs sed -i 's/from train/from .train/g'
grep -rl '^from script_train' src/nomenclatures | xargs sed -i 's/from script_train/from .script_train/g'

mkdir -p src/pipeline_siret_bi/training_classes
cp -r ../pipeline_siret_bi/training_classes/*.py  src/pipeline_siret_bi/training_classes
cp -r ../pipeline_siret_bi/elastic  src/pipeline_siret_bi/elastic
cp -r ../pipeline_siret_bi/geocodage  src/pipeline_siret_bi/geocodage
cp -r ../pipeline_siret_bi/*.py  src/pipeline_siret_bi
grep -rl '^from train' src/pipeline_siret_bi | xargs sed -i 's/from train/from .train/g'
grep -rl '^import elastic ' src/pipeline_siret_bi | xargs sed -i 's/import elastic /from . import elastic /g'
grep -rl '^import preprocessing' src/pipeline_siret_bi | xargs sed -i 's/import preprocessing/from . import preprocessing/g'
grep -rl '^from geocodage' src/pipeline_siret_bi | xargs sed -i 's/from geocodage/from .geocodage/g'
grep -rl '^from script_' src/pipeline_siret_bi | xargs sed -i 's/from script_/from .script_/g'

mkdir -p src/pipeline_bi_noncodable/training_classes
cp -r ../pipeline_bi_noncodable/training_classes/*.py  src/pipeline_bi_noncodable/training_classes
cp -r ../pipeline_bi_noncodable/*.py  src/pipeline_bi_noncodable
grep -rl '^from train' src/pipeline_bi_noncodable | xargs sed -i 's/from train/from .train/g'
grep -rl '^import preprocessing' src/pipeline_bi_noncodable | xargs sed -i 's/import preprocessing/from . import preprocessing/g'


if [ -z "$DOCKER_IMAGE_NAME" ]; then
    DOCKER_IMAGE_NAME=ssplab/aiee2-prediction/service_prediction
fi;
if [ -z "$DOCKER_IMAGE_TAG" ]; then
    DOCKER_IMAGE_TAG=latest
fi;

BUILD_ARGS=
if [[ ! -z "$PYTHON_PIP_PROXY" ]]; then
    BUILD_ARGS="--build-arg HTTP_PROXY=$PYTHON_PIP_PROXY --build-arg HTTPS_PROXY=$PYTHON_PIP_PROXY"
fi
echo "building image $DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG with args $BUILD_ARGS"
docker build $BUILD_ARGS -t "$DOCKER_IMAGE_NAME":"$DOCKER_IMAGE_TAG" .

if [[ ! -z "$DOCKER_REGISTRY" ]]; then
    if [[ ! -z "$DOCKER_LOGIN" ]]; then
        echo "login to docker registry"
        docker login -u "$DOCKER_LOGIN" -p "$DOCKER_PWD" "$DOCKER_REGISTRY"
    fi;
    echo "retagging image to $DOCKER_REGISTRY/$DOCKER_IMAGE_NAME:$DOCKER_IMAGE_TAG"
    docker tag "$DOCKER_IMAGE_NAME":"$DOCKER_IMAGE_TAG" "$DOCKER_REGISTRY"/"$DOCKER_IMAGE_NAME":"$DOCKER_IMAGE_TAG"

    echo "pushing to registry"
    docker push "$DOCKER_REGISTRY"/"$DOCKER_IMAGE_NAME":"$DOCKER_IMAGE_TAG"
fi;



# rm -rf  src/data_import
# rm -rf  src/similarity