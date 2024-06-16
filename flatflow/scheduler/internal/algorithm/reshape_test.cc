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

#include "flatflow/scheduler/internal/algorithm/reshape.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(ReshapeTest, ReshapeWithSmallTestCase) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 2);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 5);
  constexpr auto kDataParallelSize = static_cast<std::size_t>(1 << 2);
  constexpr auto kGlobalBatchSize = static_cast<std::size_t>(1 << 5);

  auto micro_batches = std::vector<std::vector<std::size_t>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t index = 0; index < kNumMicroBatches; ++index) {
    auto micro_batch = std::vector<std::size_t>(kMicroBatchSize);
    std::iota(micro_batch.begin(), micro_batch.end(), index * kMicroBatchSize);
    micro_batches.emplace_back(std::move(micro_batch));
  }

  const auto reshaped = flatflow::scheduler::internal::algorithm::reshape(
      micro_batches, kDataParallelSize, kGlobalBatchSize);

  EXPECT_EQ(reshaped.size(), kDataParallelSize);
  for (const auto &indices : reshaped) {
    EXPECT_EQ(indices.size(),
              kMicroBatchSize * kNumMicroBatches / kDataParallelSize);
  }

  const auto indices = std::vector<std::vector<std::size_t>>(
      {{0,  1,  2,  3,  4,  5,  6,  7,  32, 33, 34, 35, 36,  37,  38,  39,
        64, 65, 66, 67, 68, 69, 70, 71, 96, 97, 98, 99, 100, 101, 102, 103},
       {8,  9,  10, 11, 12, 13, 14, 15, 40,  41,  42,  43,  44,  45,  46,  47,
        72, 73, 74, 75, 76, 77, 78, 79, 104, 105, 106, 107, 108, 109, 110, 111},
       {16, 17, 18, 19, 20, 21, 22, 23, 48,  49,  50,  51,  52,  53,  54,  55,
        80, 81, 82, 83, 84, 85, 86, 87, 112, 113, 114, 115, 116, 117, 118, 119},
       {24, 25, 26,  27,  28,  29,  30,  31,  56,  57, 58,
        59, 60, 61,  62,  63,  88,  89,  90,  91,  92, 93,
        94, 95, 120, 121, 122, 123, 124, 125, 126, 127}});

  for (std::size_t rank = 0; rank < kDataParallelSize; ++rank) {
    EXPECT_TRUE(std::equal(reshaped[rank].cbegin(), reshaped[rank].cend(),
                           indices[rank].cbegin()));
  }
}

TEST(ReshapeTest, ReshapeWithLargeTestCase) {
  constexpr auto kMicroBatchSize = static_cast<std::size_t>(1 << 4);
  constexpr auto kNumMicroBatches = static_cast<std::size_t>(1 << 12);
  constexpr auto kDataParallelSize = static_cast<std::size_t>(1 << 3);
  constexpr auto kGlobalBatchSize = static_cast<std::size_t>(1 << 10);
  constexpr auto kStride = kGlobalBatchSize / kDataParallelSize;

  auto micro_batches = std::vector<std::vector<std::size_t>>();
  micro_batches.reserve(kNumMicroBatches);

  for (std::size_t index = 0; index < kNumMicroBatches; ++index) {
    auto micro_batch = std::vector<std::size_t>(kMicroBatchSize);
    std::iota(micro_batch.begin(), micro_batch.end(), index * kMicroBatchSize);
    micro_batches.emplace_back(std::move(micro_batch));
  }

  const auto reshaped = flatflow::scheduler::internal::algorithm::reshape(
      micro_batches, kDataParallelSize, kGlobalBatchSize);

  EXPECT_EQ(reshaped.size(), kDataParallelSize);
  for (const auto &indices : reshaped) {
    EXPECT_EQ(indices.size(),
              kMicroBatchSize * kNumMicroBatches / kDataParallelSize);
  }

  auto indices = std::vector<std::vector<std::size_t>>();
  indices.reserve(kDataParallelSize);

  while (indices.size() < indices.capacity()) {
    auto samples = std::vector<std::size_t>();
    samples.reserve(kNumMicroBatches / kDataParallelSize * kMicroBatchSize);
    indices.emplace_back(std::move(samples));
  }

  for (std::size_t offset = 0;
       offset < kMicroBatchSize * kNumMicroBatches / kStride; ++offset) {
    const auto rank = offset % kDataParallelSize;
    for (std::size_t index = offset * kStride; index < (offset + 1) * kStride;
         ++index) {
      indices[rank].emplace_back(index);
    }
  }

  for (std::size_t rank = 0; rank < kDataParallelSize; ++rank) {
    EXPECT_TRUE(std::equal(reshaped[rank].cbegin(), reshaped[rank].cend(),
                           indices[rank].cbegin()));
  }
}

}  // namespace
