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

#ifndef FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_RESHAPE_H_
#define FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_RESHAPE_H_

#include <algorithm>
#include <cassert>
#include <execution>
#include <iterator>
#include <utility>
#include <vector>

#include "flatflow/data/internal/types.h"

namespace flatflow {
namespace scheduler {
namespace internal {
namespace algorithm {

// reshape()
//
// Distributes the given shuffled micro-batches to each of the workers.
template <typename T>
  requires flatflow::data::internal::Unsigned<T>
std::vector<std::vector<T>> reshape(
    const std::vector<std::vector<T>> &micro_batches,
    const T &data_parallel_size, const T &global_batch_size) {
  const auto _data_parallel_size = static_cast<std::size_t>(data_parallel_size);
  const auto _global_batch_size = static_cast<std::size_t>(global_batch_size);
  assert(_data_parallel_size != 0);
  assert(_global_batch_size != 0);
  assert(_global_batch_size % _data_parallel_size == 0);

  const auto num_micro_batches = micro_batches.size();
  assert(num_micro_batches != 0);
  assert(num_micro_batches % _data_parallel_size == 0);

  const auto micro_batch_size = micro_batches[0].size();
  assert(micro_batch_size != 0);
  assert(_global_batch_size / _data_parallel_size % micro_batch_size == 0);

  // To minimize both computation stalls across pipeline stages and
  // synchronization latency between pipelines, we distribute the shuffled
  // micro-batches at the granularity of mini-batch:
  //
  // * In pipeline parallelism, all pipeline stages should have the same
  //   execution time, so micro-batches are first distributed to the same
  //   pipeline.
  // * On the other hand, in data parallelism, synchronization latency between
  //   pipelines hinders scalability (in both synchronous pipeline schedules
  //   such as GPipe and asynchronous pipeline schedules such as PipeDream),
  //   so micro-batches are then distributed to other pipelines.
  //
  // Such distribution policy that prioritizes pipeline parallelism is due to
  // the fact that computation stalls occur for each pipeline stage while
  // synchronization latency occurs only for each batch.
  const auto stride =
      _global_batch_size / _data_parallel_size / micro_batch_size;
  const auto num_samples =
      num_micro_batches / _data_parallel_size * micro_batch_size;

  auto indices = std::vector<std::vector<T>>();
  indices.reserve(_data_parallel_size);

  while (indices.size() < indices.capacity()) {
    indices.emplace_back(std::move(std::vector<T>(num_samples)));
  }

  #pragma omp parallel for
  for (std::size_t offset = 0; offset < num_micro_batches; ++offset) {
    const auto rank = offset / stride % _data_parallel_size;
    const auto index = static_cast<std::ptrdiff_t>(
        (offset / stride / _data_parallel_size * stride + offset % stride) *
        micro_batch_size);

    const auto &micro_batch = micro_batches[offset];
    std::copy(micro_batch.cbegin(), micro_batch.cend(),
              std::next(indices[rank].begin(), index));
  }

  return indices;
}

}  // namespace algorithm
}  // namespace internal
}  // namespace scheduler
}  // namespace flatflow

#endif  // FLATFLOW_SCHEDULER_INTERNAL_ALGORITHM_RESHAPE_H_
