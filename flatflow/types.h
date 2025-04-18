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

#ifndef FLATFLOW_TYPES_H_
#define FLATFLOW_TYPES_H_

#include <type_traits>

namespace flatflow {

// If the type `T` is a pointer type, provides the member typedef `type` which
// is the type pointed to by `T` with its topmost cv-qualifiers removed.
// Otherwise `type` is `T` with its topmost cv-qualifiers removed.
template <typename T>
struct remove_cvptr {
  // The type pointed by `T` or `T` itself if it is not a pointer,
  // with top-level cv-qualifiers removed.
  using type = std::remove_cv_t<std::remove_pointer_t<T>>;
};

// Alias template for `remove_cvptr`.
template <typename T>
using remove_cvptr_t = typename remove_cvptr<T>::type;

// The concept `arithmetic<T>` is satisfied if and only if `T` is an arithmetic
// type (that is, an integral type or a floating-point type). This is intended
// as a supplementary concept to the standard concepts library, filling the gap
// of a concept for arithmetic types the standard lacks.
template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

// The concept `scalar<T>` is satisfied if and only if `T` is a scalar type.
// This is intended as a supplementary concept to the standard concepts library,
// filling the gap of a concept for scalar types the standard lacks.
template <typename T>
concept scalar = std::is_scalar_v<T>;

}  // namespace flatflow

#endif  // FLATFLOW_TYPES_H_
