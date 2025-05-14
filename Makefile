# Set POD_NAME to the username if not already set
POD_NAME ?= $(shell whoami)

###########
# Commands#
###########

# Create directories and run the Ansible playbook with the specified environment variables
run-local:
	mkdir -p htmlcov
	ansible-playbook embedding_playbook.yml \
	-e "container_state=started" \
	-e "use_gpus=true"

run-github:
	mkdir -p htmlcov
	ansible-playbook embedding_playbook.yml \
	-e "container_state=started" \
	-e "use_gpus=false" \
	-e "start_github_only=true"

# Stop the Ansible playbook and set the container state to 'absent'
stop:
	@echo "Stopping the playbook..."
	@export POD_NAME=$(POD_NAME) && \
	ansible-playbook embedding_playbook.yml \
	-e "container_state=absent"

# Stop the Ansible playbook and set the container state to 'absent'
stop-local:
	@echo "Stopping the playbook..."
	@export POD_NAME=$(POD_NAME) && \
	ansible-playbook embedding_playbook.yml \
	-e "container_state=absent" \
	-e "use_gpus=true"

test-embeddings:
	python3 experiments/embedding_tests.py --method triplet --pod-name $(POD_NAME)

# Define the container name based on the POD_NAME
CONTAINER_NAME := $(POD_NAME)_dev

###########
# Container Check #
###########

# Check if the Podman container exists
dev-exists:
	@echo "Checking that the container $(CONTAINER_NAME) exists..."
	@podman container exists $(CONTAINER_NAME) || \
	(echo "Container $(CONTAINER_NAME) does not exist. Please create it." && exit 1)
	@echo "Container $(CONTAINER_NAME) exists."

# Check if the Podman container is running
dev-running:
	@echo "Checking that the container $(CONTAINER_NAME) is running..."
	@podman container exists $(CONTAINER_NAME) || \
	(echo "Container $(CONTAINER_NAME) does not exist. Please create it." && exit 1)
	@podman container inspect $(CONTAINER_NAME) | \
	jq -r '.[0].State.Status' | grep running || \
	(echo "Container $(CONTAINER_NAME) is not running. Please start it." && exit 1)
	@echo "Container $(CONTAINER_NAME) is running."

###########
# Quality Checks #
###########

# Run ansible-lint inside the container
ansible-lint:
	@echo "Running ansible-lint..."
	@podman exec $(CONTAINER_NAME) ansible-lint

# Build documentation using Sphinx
build-docs:
	podman exec $(CONTAINER_NAME) sphinx-build docs/source docs/build

# Code quality checks
## Check code formatting
check-format-code:
	podman exec $(CONTAINER_NAME) ruff check finetuning_embeddings

## Format code
format-code:
	podman exec $(CONTAINER_NAME) ruff check finetuning_embeddings --fix

## Type check
type-check:
	podman exec $(CONTAINER_NAME) mypy finetuning_embeddings --ignore-missing-imports

# Format and type check
check: format-code type-check

# Run tests
test:
	podman exec $(CONTAINER_NAME) python -m pytest

# Run tests with coverage report
test-coverage:
	podman exec $(CONTAINER_NAME) python -m pytest --cov --cov-report=html

#########
# Setup #
#########

# Verify necessary software is installed
verify-software:
	@echo "The shell being used is:"
	@echo $(shell echo $$SHELL)
	@echo "Checking if Podman is installed..."
	podman --version
	@echo "Checking if Python is installed..."
	python --version

# Install pre-commit hooks
install-precommit:
	pip install pre-commit
	pre-commit install

# Setup environment
setup: verify-software install-precommit
	@echo "You are ready to go!"
