#!/usr/bin/env bash

########################
#  Build docker
#######################
set -e

mkdir -p src/data_import
cp -r ../data_import/*.py  src/data_import

mkdir -p src/config
cp -r ../config/*.py  src/config

mkdir -p src/nomenclatures/training_classes
cp -r ../nomenclatures/training_classes/*.py  src/nomenclatures/training_classes
cp -r ../nomenclatures/*.py  src/nomenclatures
grep -rl '^from train' src/nomenclatures | xargs sed -i 's/from train/from .train/g'
grep -rl '^from script_train' src/nomenclatures | xargs sed -i 's/from script_train/from .script_train/g'

mkdir -p src/pipeline_siret_bi/training_classes
cp -r ../pipeline_siret_bi/training_classes/*.py  src/pipeline_siret_bi/training_classes
cp -rT ../pipeline_siret_bi/elastic  src/pipeline_siret_bi/elastic
cp -rT ../pipeline_siret_bi/geocodage  src/pipeline_siret_bi/geocodage
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
    DOCKER_IMAGE_NAME=ssplab/aiee2/service_prediction
fi;
if [ -z "$CI_COMMIT_TAG" ]; then
    CI_COMMIT_TAG=latest
fi;

BUILD_ARGS=
if [[ ! -z "$PYTHON_PIP_PROXY" ]]; then
    BUILD_ARGS="--build-arg HTTP_PROXY=$PYTHON_PIP_PROXY --build-arg HTTPS_PROXY=$PYTHON_PIP_PROXY"
fi

mkdir -p /kaniko/.docker
echo "{\"auths\":{\"$DOCKER_REGISTRY\":{\"username\":\"robot\$ssplab+gitlab\",\"password\":\"$REGISTRY_PASSWORD\"}}}" > /kaniko/.docker/config.json
echo "building image $DOCKER_IMAGE_NAME:$CI_COMMIT_TAG with args $BUILD_ARGS"
/kaniko/executor --context $CI_PROJECT_DIR/service_prediction --skip-tls-verify --registry-mirror $DOCKER_REGISTRY --dockerfile $CI_PROJECT_DIR/service_prediction/Dockerfile_app --destination $DOCKER_REGISTRY/$DOCKER_IMAGE_NAME:$CI_COMMIT_TAG --single-snapshot
