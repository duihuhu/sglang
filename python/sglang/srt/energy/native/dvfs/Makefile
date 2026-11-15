# Build packaged NVML sources without initializing NVML or changing GPU clocks.
# Example: make OUTPUT=/tmp/libdvfs_ctrl.so NVML_INCLUDE_DIR=/usr/local/cuda/include
PYTHON ?= python3
SOURCE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
XDG_CACHE_HOME ?= $(HOME)/.cache
OUTPUT ?= $(XDG_CACHE_HOME)/sglang/dvfs/libdvfs_ctrl.so
CXXFLAGS ?= -O2 -Wall
CPPFLAGS ?=
LDFLAGS ?=
CUDA_HOME ?= $(if $(wildcard /usr/local/cuda/include/nvml.h),/usr/local/cuda)
NVML_INCLUDE_DIR ?= $(if $(CUDA_HOME),$(CUDA_HOME)/include)
NVML_LIB_DIR ?=

.PHONY: all clean help
all: $(OUTPUT)

$(OUTPUT): $(SOURCE_DIR)dvfs_ctrl.cpp
	mkdir -p "$(dir $(OUTPUT))"
	$(CXX) -shared -fPIC -std=c++11 $(CPPFLAGS) $(CXXFLAGS) $(if $(NVML_INCLUDE_DIR),-I"$(NVML_INCLUDE_DIR)") "$<" -o "$@" $(if $(NVML_LIB_DIR),-L"$(NVML_LIB_DIR)") $(LDFLAGS) -lnvidia-ml

clean:
	rm -f "$(OUTPUT)"

help:
	@echo 'make OUTPUT=/absolute/path/libdvfs_ctrl.so [CUDA_HOME=...] [NVML_INCLUDE_DIR=...] [NVML_LIB_DIR=...]'
	@echo 'Alternatively: $(PYTHON) -m sglang.srt.energy.native.dvfs.build --help'
