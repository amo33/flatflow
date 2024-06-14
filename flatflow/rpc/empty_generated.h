// automatically generated by the FlatBuffers compiler, do not modify


#ifndef FLATBUFFERS_GENERATED_EMPTY_FLATFLOW_RPC_H_
#define FLATBUFFERS_GENERATED_EMPTY_FLATFLOW_RPC_H_

#include "flatbuffers/flatbuffers.h"

// Ensure the included flatbuffers.h is the same version as when this file was
// generated, otherwise it may not be compatible.
static_assert(FLATBUFFERS_VERSION_MAJOR == 24 &&
              FLATBUFFERS_VERSION_MINOR == 3 &&
              FLATBUFFERS_VERSION_REVISION == 25,
             "Non-compatible flatbuffers version included");

namespace flatflow {
namespace rpc {

struct Empty;
struct EmptyBuilder;

/// A generic empty message that you can reuse to avoid defining duplicated
/// empty messages in your APIs. A typical example is to use it as the request
/// or the response type of an API method.
struct Empty FLATBUFFERS_FINAL_CLASS : private ::flatbuffers::Table {
  typedef EmptyBuilder Builder;
  bool Verify(::flatbuffers::Verifier &verifier) const {
    return VerifyTableStart(verifier) &&
           verifier.EndTable();
  }
};

struct EmptyBuilder {
  typedef Empty Table;
  ::flatbuffers::FlatBufferBuilder &fbb_;
  ::flatbuffers::uoffset_t start_;
  explicit EmptyBuilder(::flatbuffers::FlatBufferBuilder &_fbb)
        : fbb_(_fbb) {
    start_ = fbb_.StartTable();
  }
  ::flatbuffers::Offset<Empty> Finish() {
    const auto end = fbb_.EndTable(start_);
    auto o = ::flatbuffers::Offset<Empty>(end);
    return o;
  }
};

inline ::flatbuffers::Offset<Empty> CreateEmpty(
    ::flatbuffers::FlatBufferBuilder &_fbb) {
  EmptyBuilder builder_(_fbb);
  return builder_.Finish();
}

}  // namespace rpc
}  // namespace flatflow

#endif  // FLATBUFFERS_GENERATED_EMPTY_FLATFLOW_RPC_H_