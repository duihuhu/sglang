cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(afd_ipc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Torch REQUIRED)

# pybind11 (use torch's bundled pybind11)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find pybind11
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
)
if(NOT PYBIND11_RESULT EQUAL 0)
    # Fallback: use torch's pybind11
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])"
        OUTPUT_VARIABLE TORCH_INCLUDE
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()
find_package(pybind11 CONFIG)

# Source files
set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sgl-kernel/csrc/afd_ipc/afd_ipc.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sgl-kernel/csrc/afd_ipc/afd_ipc_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp
)

# Build the Python extension module
pybind11_add_module(afd_ipc_cpp ${SOURCES})

target_include_directories(afd_ipc_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../sgl-kernel/csrc/afd_ipc
    ${TORCH_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(afd_ipc_cpp PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    pthread
    rt
)

# CUDA architecture (match the system)
set_property(TARGET afd_ipc_cpp PROPERTY CUDA_ARCHITECTURES "80;86;89;90")

# Install to the Python package directory
install(TARGETS afd_ipc_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
