.MAKEFLAGS += --warn-undifined-variables --no-print-directory
.SHELLFALGS := -ue -o pipfail -c

all: help
.PHONY: all

# Use bash for inline if-statements
SHELL:=bash
APP_NAME=$(shell basename "`pwd`")
OWNER?=martin.krause
DOCKER_REPOSITORY=local
SOURCE_IMAGE=$(DOCKER_REPOSITORY)/$(OWNER)/$(APP_NAME)

# Enable BuildKit for Docker build
export DOCKER_BUILDKIT:=1

##@ Helpers
help: ## display this help
	@echo "$(APP_NAME)"
	@echo "============================="
	@awk 'BEGIN {FS = ":.*##"; printf "\033[36m\033[0m"} /^[a-zA-Z0-9_%\/-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
	@printf "\n"

##@ Build Jupyterlab Image
build: IMAGE_TAG?=latest
build: ## build images
	docker build --rm --force-rm -t $(SOURCE_IMAGE):$(IMAGE_TAG) .


##@ Run Jupyterlab container
run: DARGS?=-e JUPYTER_ENABLE_LAB=yes
run: IMAGE_TAG?=latest
run: PORT?=8888
run: JARGS?=start-notebook.sh --NotebookApp.token="" --NotebookApp.notebook_dir=/home/jovyan/work
run: CONTAINER_NAME=$(APP_NAME)-jupyterlab
run: ## Run container
	docker run \
	-it --rm -p $(PORT):8888 --name $(CONTAINER_NAME) \
	-e PYTHONPATH=$(PYTHONPATH):/home/jovyan/work/src \
	-e DATA_DIR=/home/jovyan/work/data \
	-v $(shell pwd)/..:/home/jovyan/work $(DARGS) \
	$(SOURCE_IMAGE):$(IMAGE_TAG) $(JARGS)

##@ Run Jupyterlab + MLflow tracking URI container using docker-compose
run-jup-mlflow: ## Run Jupyter and Mlflow UI
	docker-compose up --build
