# Copyright 2024 The FlatFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(pybind11 REQUIRED)
find_package(Python3 COMPONENTS Development REQUIRED)

find_package(Protobuf REQUIRED)

file(GLOB SOURCE_FILE "${CMAKE_SOURCE_DIR}/flatflow/rpc/*.fb.cc")
file(GLOB HEADER_FILE "${CMAKE_SOURCE_DIR}/flatflow/rpc/*.h")
file(GLOB PYTHON_FILE "${CMAKE_SOURCE_DIR}/flatflow/python/*.cpp" "${CMAKE_SOURCE_DIR}/flatflow/python/*.h")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

if(NOT SOURCE_FILE AND NOT PYTHON_FILE)
    message(FATAL_ERROR "No source files found for target 'rpc'. Please check your directory structure and file names.")
endif()

pybind11_add_module(rpc 
    ${SOURCE_FILE}
    ${HEADER_FILE}
    ${PYTHON_FILE}
)

target_include_directories(rpc 
    PRIVATE ${CMAKE_SOURCE_DIR}
)

target_link_libraries(rpc PRIVATE
    absl::btree
    absl::inlined_vector
    absl::log
    absl::log_initialize
    absl::str_format
    absl::flags_usage
    BLAS::BLAS
    pybind11::module
    grpc++
    grpc++_reflection
    flatbuffers
    OpenMP::OpenMP_CXX
)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_LIBRARY_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS rpc
  COMPONENT python
  LIBRARY DESTINATION "${PYTHON_LIBRARY_DIR}"
)
