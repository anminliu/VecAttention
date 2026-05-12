.PHONY: help setup-host setup-container download-models download-datasets faclean fainstall lmdeploy ditinit vlminit

REPO_ROOT := $(shell cd "$(dir $(abspath $(lastword $(MAKEFILE_LIST))))" && pwd)

# Default parameters
PROXY ?=
MODELS ?=
DATASETS ?=
DOWNLOAD_PATH ?= ./

# Docker settings (override on command line if desired)
DOCKER_IMAGE ?= vecattention
DOCKER_CONTAINER ?= vecattention
# Directory on the host to mount as /workspace inside the container.
# Defaults to the parent of the repo root so that the repo appears at
# /workspace/VecAttention inside the container.
HOST_WORKSPACE ?= $(dir $(REPO_ROOT))

# lmdeploy
CUDA_VISIBLE_DEVICES ?= 0
LMDEPLOY_MODEL_PATH ?= /workspace/models/<your-model>

setup-host:
	docker build \
		$(if $(PROXY),--build-arg PROXY="$(PROXY)") \
		-t $(DOCKER_IMAGE) -f docker/dockerfile . && \
	docker run -dit --gpus all --ipc=host \
		-v $(HOST_WORKSPACE):/workspace \
		--privileged --network=host --name $(DOCKER_CONTAINER) $(DOCKER_IMAGE) bash

setup-container:
	git config --global --add safe.directory $(REPO_ROOT) && \
	cd $(REPO_ROOT) && \
	$(if $(PROXY),export http_proxy="$(PROXY)" https_proxy="$(PROXY)" HTTP_PROXY="$(PROXY)" HTTPS_PROXY="$(PROXY)" &&) \
	uv venv && \
	uv pip install --no-config torch==2.7.0 torchcodec==0.5 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
	uv pip install --no-config setuptools Cython wheel packaging pybind11 flashinfer-python==0.3.1.post1 && \
	uv pip install --no-config triton@git+https://github.com/openai/triton.git@6e390f3f
	source $(REPO_ROOT)/.venv/bin/activate && \
	echo "Container environment setup complete."

lmdeploy:
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) uvx lmdeploy serve api_server $(LMDEPLOY_MODEL_PATH) --server-port 23333

faclean:
	uv pip uninstall vllm-flash-attn && \
	uv cache clean vllm-flash-attn && \
	rm -rf $(REPO_ROOT)/vllm-flash-attention/build

fainstall:
	$(if $(PROXY),export http_proxy="$(PROXY)" https_proxy="$(PROXY)" HTTP_PROXY="$(PROXY)" HTTPS_PROXY="$(PROXY)" &&) \
	uv pip install --no-config psutil && \
	cd $(REPO_ROOT)/vllm-flash-attention && \
	uv run --no-sync --no-project --no-build-isolation -- python setup.py install

ditinit:
	cd $(REPO_ROOT) && \
	git submodule update --init --recursive && \
	uv pip install --no-config setuptools Cython wheel packaging cmake ninja && \
	uv pip install --no-config torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
	uv sync --no-build-isolation --group dit && \
	cd $(REPO_ROOT)/eval/DiTEvalKit/kernels && \
	bash setup.sh && \
	FLASHINF_DIR=$(REPO_ROOT)/eval/DiTEvalKit/kernels/3rdparty/flashinfer && \
	PATCH_FILE=$(REPO_ROOT)/eval/DiTEvalKit/kernels/modifications.patch && \
	{ \
	  if git -C $$FLASHINF_DIR apply --check $$PATCH_FILE 2>/dev/null; then \
	    echo "[patch] applying modifications.patch"; \
	    git -C $$FLASHINF_DIR apply --3way --whitespace=fix $$PATCH_FILE; \
	  else \
	    echo "[patch] already applied or context changed, skipping"; \
	  fi; \
	} && \
	cd $$FLASHINF_DIR && \
	uv pip install --no-config --no-build-isolation --verbose --editable . && \
	uv pip install --no-config cuvs-cu12 --extra-index-url=https://pypi.nvidia.com

vlminit:
	cd $(REPO_ROOT) && \
	git submodule update --init --recursive && \
	uv pip install --no-config setuptools Cython wheel packaging cmake ninja && \
	uv pip install --no-config torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
	MAX_JOBS=16 uv sync --no-build-isolation --group vlm
