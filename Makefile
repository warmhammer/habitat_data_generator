SHELL	:=	/bin/bash

MKFILE_PATH	:=	$(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR 	:= 	$(dir $(MKFILE_PATH))
ROOT_DIR 	:= 	$(MKFILE_DIR)

DOCKER_COMPOSE_FILE := \
	-f docker/docker-compose.yml

DATA_DIR 	?= 	$(ROOT_DIR)/../data
PARAMETERS	:= 	ROOT_DIR=$(ROOT_DIR) \
				DATA_DIR=$(DATA_DIR)

prepare:
	@ xhost +local:docker

build-habitat:
	@ cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose $(DOCKER_COMPOSE_FILE) build habitat 

up-habitat:
	@ cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose $(DOCKER_COMPOSE_FILE) up habitat -d

into-habitat:
	@ cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose $(DOCKER_COMPOSE_FILE) exec habitat bash

stop-habitat:
	@ cd $(ROOT_DIR) && \
	$(PARAMETERS) \
	docker compose $(DOCKER_COMPOSE_FILE) stop habitat
		