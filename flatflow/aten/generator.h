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

#ifndef FLATFLOW_ATEN_GENERATOR_H_
#define FLATFLOW_ATEN_GENERATOR_H_

#include <cstdint>

#include "ATen/core/MT19937RNGEngine.h"

namespace flatflow {
namespace aten {

// An STL-compatible wrapper class for `at::Generator`.
class Generator {
 public:
  using result_type = uint64_t;

  // Constructors and assignment operators
  //
  // Constructs a Mersenne Twister engine object, and initializes its internal
  // state sequence to pseudo-random values.
  //
  // Unlike `at::Generator`, this generator class provides copy and move
  // constructors and assignment operators to allow copy elision while
  // satisfying the `UniformRandomBitGenerator`.
  explicit Generator(result_type seed = 5489) { engine_ = at::mt19937(seed); }

  explicit Generator(const Generator &other) = default;

  Generator &operator=(const Generator &other) = default;

  explicit Generator(Generator &&other) = default;

  Generator &operator=(Generator &&other) = default;

  // Generator::min()
  //
  // Returns the minimum value potentially generated by the underlying engine.
  static constexpr result_type min() noexcept { return 0; }

  // Generator::max()
  //
  // Returns the maximum value potentially generated by the underlying engine.
  static constexpr result_type max() noexcept { return UINT32_MAX; }

  // Generator::operator()()
  //
  // Advances the state of the underlying engine, and generates a pseudo-random
  // number from the new state.
  inline result_type operator()() {
    return static_cast<result_type>(engine_());
  }

 protected:
  at::mt19937 engine_;
};

}  // namespace aten
}  // namespace flatflow

#endif  // FLATFLOW_ATEN_GENERATOR_H_
