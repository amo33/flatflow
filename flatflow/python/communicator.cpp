// Copyright 2024 The FlatFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "flatflow/rpc/communicator.h"
#include "flatflow/rpc/communicator.grpc.fb.h"

namespace py = pybind11;

PYBIND11_MODULE(rpc, m) {
  py::class_<flatflow::rpc::CommunicatorServiceImpl>(m, "Communicator")
      .def(py::init<>())
      .def("init", &flatflow::rpc::CommunicatorServiceImpl::Init)
      .def("broadcast", &flatflow::rpc::CommunicatorServiceImpl::Broadcast)
      .def("finalize", &flatflow::rpc::CommunicatorServiceImpl::Finalize);
}
