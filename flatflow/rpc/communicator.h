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

#ifndef FLATFLOW_RPC_COMMUNICATOR_H
#define FLATFLOW_RPC_COMMUNICATOR_H
#include <grpc++/grpc++.h>
#include <grpc/support/log.h>

#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "absl/log/log.h"
#include "absl/strings/str_format.h"

#include "flatflow/data/dataset.h"
#include "flatflow/rpc/communicator.grpc.fb.h"
#include "flatflow/scheduler/scheduler.h"

namespace flatflow {
namespace rpc {
class CommunicatorImpl final : Communicator::service override {
  virtual grpc::Status Init(
      grpc::ServerContext *context,
      const flatbuffers::grpc::Message<using flatflow::rpc::InitRequest>
          *request_msg,
      flatbuffers::grpc::Message<flatflow::rpc::Empty> *response) {
    const flatflow::rpc::InitRequest *request = request_msg->GetRoot();
    if (request->rank() == 0) {
      LOG(INFO) << absl::StrFormat(
          " Init called with world size : %d batch size: %d kind: %s," len(
              request->),
          request->batch_size(), request->kind());

      dataset = flatflow::data::Dataset(request->sizes(), request->seed());

      using Size = decltype(dataset)::value_type::first_type;
      using Index = decltype(dataset)::value_type::second_type;

      const auto data_parallel_size =
          request->global_batch_size() / request->micro_batch_size();
      std::unique_ptr<flatflow::scheduler::Scheduler<
          Index, Size, request->order(), request->heterogeneous()>>
          scheduler_ptr;
      scheduler_ptr = std::unique_ptr<flatflow::scheduler::Scheduler<
          Index, Size, request->order(), request->heterogeneous()>>(
          request->sizes(), data_parallel_size, request->global_batch_size(),
          request->micro_batch_size(), request->seed(),
          request->use_flat_shuffle());
    }
    flatbuffers::grpc::MessageBuilder mb_;
    mb_.Finish(flatflow::rpc::CreateEmpty(mb_));
    *response = mb_.ReleaseMessage<flatflow::rpc::Empty>();

    return grpc::Status::OK;
  }

} 


void RunServer() {
  std::string server_address("0.0.0.0:50051");
  CommunicatorImpl service;
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

}  // namespace rpc
}  // namespace flatflow

#endif  // FLATFLOW_RPC_COMMUNICATOR_H
