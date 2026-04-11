# Build libdvfs_ctrl.so from dvfs_ctrl.cpp
#
# Prerequisites:
#   - CUDA toolkit installed (for nvml.h)
#   - nvidia-ml library available (libnvidia-ml.so)
#
# Usage:
#   make            # build .so
#   make clean      # remove .so
#   make test       # quick smoke test

CXX      ?= g++
CXXFLAGS := -shared -fPIC -O2 -Wall
LDFLAGS  := -lnvidia-ml

# Auto-detect CUDA include path
CUDA_HOME ?= $(shell [ -d /usr/local/cuda ] && echo /usr/local/cuda || echo /usr/lib/cuda)
NVML_INC  := -I$(CUDA_HOME)/include

TARGET := libdvfs_ctrl.so
SRC    := dvfs_ctrl.cpp

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(NVML_INC) -o $@ $< $(LDFLAGS)
	@echo "[OK] Built $(TARGET)"

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@python3 -c "\
import ctypes; \
lib = ctypes.CDLL('./$(TARGET)'); \
assert lib.dvfs_init() == 0, 'nvmlInit failed'; \
n = lib.dvfs_get_device_count(); \
print(f'[OK] NVML initialized, {n} GPU(s) detected'); \
lib.dvfs_shutdown(); \
"