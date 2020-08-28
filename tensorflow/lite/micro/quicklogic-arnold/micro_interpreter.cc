/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/tensor_utils.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/c/builtin_op_data.h"

void PrintTensor(TfLiteTensor* ptensor, int itensor, int inode, int itensorx, bool fPrintData) {
  printf("tensor[%d]->bytes=%d ", itensor, ptensor->bytes);
  switch (ptensor->type) {
    case 2: printf("Int32"); break;
    case 3: printf("Uint8"); break;
    default: printf(" type=%d", ptensor->type);
  }

  if (ptensor->allocation_type == kTfLiteMmapRo) printf(" RO\n");
  if (ptensor->allocation_type == kTfLiteArenaRw) printf(" RW\n");
  
  printf("QuantizationType=%s\n", (ptensor->quantization.type == kTfLiteAffineQuantization) ? "affine" : "none");
  
  printf("scale=%f, zero_point=%d\n", ptensor->params.scale, ptensor->params.zero_point);
  for (size_t k = 0; k != ptensor->dims->size; k++) {
    printf("  %d", ptensor->dims->data[k]);
  }
  printf("\n");
  if (ptensor->type == 3) {
    int uxmax = 0, uxmin = 1000;
    for (int i = 0; i != ptensor->bytes; i++) {
      //printf("%d\n", ptensor->data.uint8[i]);
      if (ptensor->data.uint8[i] < uxmin) uxmin = ptensor->data.uint8[i];
      if (ptensor->data.uint8[i] > uxmax) uxmax = ptensor->data.uint8[i];
    }
    printf("Range: [%d,%d] => [%d,%d]\n", uxmin, uxmax, uxmin - ptensor->params.zero_point, uxmax - ptensor->params.zero_point);
  }
  
  if (fPrintData && ((ptensor->type & 0xFF) == 2)) {
    printf("int32_t aucNode%dTensor%d[] = {\n", inode, itensor);
    int idata = 0;
    while (idata < (ptensor->bytes/4)) {
      printf("  ");
      for (int i = 0; i != 8 && idata < ptensor->bytes; i++) {
        printf("%8d, ", ptensor->data.i32[idata++]);
      }
      printf("\n");
    }
  }
  if (fPrintData && ((ptensor->type & 0xFF) == 3)) {
    printf("uint8_t aucNode%dTensor%d[] = {\n", inode, itensor);
    int idata = 0;
    while (idata < ptensor->bytes) {
      printf("  ");
      for (int i = 0; i != 16 && idata < ptensor->bytes; i++) {
        printf("0x%02x, ", ptensor->data.uint8[idata++]);
      }
      printf("\n");
    }
  }
}

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};



void PrintNode(TfLiteNode* pnode) {
  printf("user_data=0x%x\n", pnode->user_data);
  printf("builtin_data=0x%x\n", pnode->builtin_data);
  printf("custom_initial_data=0x%x\n", pnode->custom_initial_data);
  
  const OpData* popdata = (static_cast<const OpData*>(pnode->user_data));
  printf("output_multiplier=0x%x\n", popdata->output_multiplier);
  printf("output_shift=%d\n", popdata->output_shift);
  printf("per_channel_output_multiplier=ox%x\n", popdata->per_channel_output_multiplier);
  printf("per_channel_output_shift=0x%x\n", popdata->per_channel_output_shift);
}

namespace tflite {
namespace {

const char* OpNameFromRegistration(const TfLiteRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

namespace internal {

TfLiteStatus ContextHelper::AllocatePersistentBuffer(TfLiteContext* ctx,
                                                     size_t bytes, void** ptr) {
  return reinterpret_cast<ContextHelper*>(ctx->impl_)
      ->allocator_->AllocatePersistentBuffer(bytes, ptr);
}

TfLiteStatus ContextHelper::RequestScratchBufferInArena(TfLiteContext* ctx,
                                                        size_t bytes,
                                                        int* buffer_idx) {
  ContextHelper* helper = reinterpret_cast<ContextHelper*>(ctx->impl_);
  return helper->allocator_->RequestScratchBufferInArena(
      helper->current_node_idx_, bytes, buffer_idx);
}

void* ContextHelper::GetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  return reinterpret_cast<ContextHelper*>(ctx->impl_)
      ->allocator_->GetScratchBuffer(buffer_idx);
}

void ContextHelper::ReportOpError(struct TfLiteContext* context,
                                  const char* format, ...) {
  ContextHelper* helper = static_cast<ContextHelper*>(context->impl_);
  va_list args;
  va_start(args, format);
  TF_LITE_REPORT_ERROR(helper->error_reporter_, format, args);
  va_end(args);
}

}  // namespace internal

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   ErrorReporter* error_reporter,
                                   tflite::Profiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*MicroAllocator::Create(tensor_arena, tensor_arena_size,
                                         error_reporter)),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      context_helper_(error_reporter_, &allocator_) {
  Init(profiler);
}

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   MicroAllocator* allocator,
                                   ErrorReporter* error_reporter,
                                   tflite::Profiler* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      error_reporter_(error_reporter),
      allocator_(*allocator),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      context_helper_(error_reporter_, &allocator_) {
  Init(profiler);
}

MicroInterpreter::~MicroInterpreter() {
  if (node_and_registrations_ != nullptr) {
    for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
      TfLiteNode* node = &(node_and_registrations_[i].node);
      const TfLiteRegistration* registration =
          node_and_registrations_[i].registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(&context_, node->user_data);
      }
    }
  }
}

void MicroInterpreter::Init(tflite::Profiler* profiler) {
  const flatbuffers::Vector<flatbuffers::Offset<SubGraph>>* subgraphs =
      model_->subgraphs();
  if (subgraphs->size() != 1) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Only 1 subgraph is currently supported.\n");
    initialization_status_ = kTfLiteError;
    return;
  }
  subgraph_ = (*subgraphs)[0];

  context_.impl_ = static_cast<void*>(&context_helper_);
  context_.ReportError = context_helper_.ReportOpError;
  context_.recommended_num_threads = 1;
  context_.profiler = profiler;

  initialization_status_ = kTfLiteOk;
}

void MicroInterpreter::CorrectTensorEndianness(TfLiteTensor* tensorCorr) {
  int32_t tensorSize = 1;
  for (int d = 0; d < tensorCorr->dims->size; ++d)
    tensorSize *= reinterpret_cast<const int32_t*>(tensorCorr->dims->data)[d];

  switch (tensorCorr->type) {
    case TfLiteType::kTfLiteFloat32:
      CorrectTensorDataEndianness(tensorCorr->data.f, tensorSize);
      break;
    case TfLiteType::kTfLiteFloat16:
      CorrectTensorDataEndianness(tensorCorr->data.f16, tensorSize);
      break;
    case TfLiteType::kTfLiteInt64:
      CorrectTensorDataEndianness(tensorCorr->data.i64, tensorSize);
      break;
    case TfLiteType::kTfLiteInt32:
      CorrectTensorDataEndianness(tensorCorr->data.i32, tensorSize);
      break;
    case TfLiteType::kTfLiteInt16:
      CorrectTensorDataEndianness(tensorCorr->data.i16, tensorSize);
      break;
    case TfLiteType::kTfLiteComplex64:
      CorrectTensorDataEndianness(tensorCorr->data.c64, tensorSize);
      break;
    default:
      // Do nothing for other data types.
      break;
  }
}

template <class T>
void MicroInterpreter::CorrectTensorDataEndianness(T* data, int32_t size) {
  for (int32_t i = 0; i < size; ++i) {
    data[i] = flatbuffers::EndianScalar(data[i]);
  }
}

TfLiteStatus MicroInterpreter::AllocateTensors() {
  if (allocator_.StartModelAllocation(model_, &context_, op_resolver_,
                                      &node_and_registrations_) != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Failed starting model allocation.\n");
    initialization_status_ = kTfLiteError;
    return kTfLiteError;
  }

  // If the system is big endian then convert weights from the flatbuffer from
  // little to big endian on startup so that it does not need to be done during
  // inference.
  // NOTE: This requires that the flatbuffer is held in memory which can be
  // modified by this process.
  if (!FLATBUFFERS_LITTLEENDIAN) {
    for (size_t t = 0; t < tensors_size(); ++t) {
      TfLiteTensor* thisTensor = &context_.tensors[t];
      if (thisTensor->allocation_type == kTfLiteMmapRo)
        CorrectTensorEndianness(thisTensor);
    }
  }

  // Only allow AllocatePersistentBuffer in Init stage.
  context_.AllocatePersistentBuffer = context_helper_.AllocatePersistentBuffer;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = nullptr;

  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    context_helper_.SetNodeIndex(i);
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;
    size_t init_data_size;
    const char* init_data;
    if (registration->builtin_code == BuiltinOperator_CUSTOM) {
      init_data = reinterpret_cast<const char*>(node->custom_initial_data);
      init_data_size = node->custom_initial_data_size;
    } else {
      init_data = reinterpret_cast<const char*>(node->builtin_data);
      init_data_size = 0;
    }
    if (registration->init) {
      node->user_data =
          registration->init(&context_, init_data, init_data_size);
    }
  }
  context_helper_.SetNodeIndex(-1);

  // Both AllocatePersistentBuffer and RequestScratchBufferInArena is available
  // in Prepare stage.
  context_.RequestScratchBufferInArena =
      context_helper_.RequestScratchBufferInArena;
  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    // Set node idx to annotate the lifetime for scratch buffers.
    context_helper_.SetNodeIndex(i);
    auto* node = &(node_and_registrations_[i].node);
    auto* registration = node_and_registrations_[i].registration;
    if (registration->prepare) {
      TfLiteStatus prepare_status = registration->prepare(&context_, node);
      if (prepare_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Node %s (number %df) failed to prepare with status %d",
            OpNameFromRegistration(registration), i, prepare_status);
        return kTfLiteError;
      }
    }
  }
  context_helper_.SetNodeIndex(-1);

  // Prepare is done, we're ready for Invoke. Memory allocation is no longer
  // allowed. Kernels can only fetch scratch buffers via GetScratchBuffer.
  context_.AllocatePersistentBuffer = nullptr;
  context_.RequestScratchBufferInArena = nullptr;
  context_.GetScratchBuffer = context_helper_.GetScratchBuffer;

  TF_LITE_ENSURE_OK(&context_,
                    allocator_.FinishModelAllocation(model_, &context_));
  tensors_allocated_ = true;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::Invoke() {
  if (initialization_status_ != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Invoke() called after initialization failed\n");
    return kTfLiteError;
  }

  // Ensure tensors are allocated before the interpreter is invoked to avoid
  // difficult to debug segfaults.
  if (!tensors_allocated_) {
    TF_LITE_ENSURE_OK(&context_, AllocateTensors());
  }

  for (size_t i = 0; i < subgraph_->operators()->size(); ++i) {
    if (i == 3) break;
    printf("\n\nWorking on node[%d]\n", i);
    auto* node = &(node_and_registrations_[i].node);

    
    auto* registration = node_and_registrations_[i].registration;
    if ((registration->builtin_code == 3)  || i == 27) {
      if ((i<100) && (registration->builtin_code == 3)) {
        context_.tensors[node->inputs->data[0]].type |= 0x500;  // Flag as limited coeff & print output
      }
    
          PrintNode(node);
          
      switch(registration->builtin_code) {
          case 3: printf("BuiltinConv2d\n"); break;
          case 4: printf("BuiltinDepthwiseConv2d\n"); break;
          default:
              printf("registration->builtin_code = %d\n", registration->builtin_code);
      }
      for (size_t j = 0; j != node->inputs->size; j++) {
        PrintTensor(&context_.tensors[node->inputs->data[j]], node->inputs->data[j], i, j, false);
      }
      for (size_t j = 0; j != node->outputs->size; j++) {
        PrintTensor(&context_.tensors[node->outputs->data[j]], node->outputs->data[j], i, j, false);
      }
    }

    if (registration->invoke) {
      TfLiteStatus invoke_status;
#ifndef NDEBUG  // Omit profiler overhead from release builds.
      // The case where profiler == nullptr is handled by ScopedOperatorProfile.
      tflite::Profiler* profiler =
          reinterpret_cast<tflite::Profiler*>(context_.profiler);
      ScopedOperatorProfile scoped_profiler(
          profiler, OpNameFromRegistration(registration), i);
#endif
      printf("invoking\n");
      printf("int32_t axNode%dTensor%d[] = {\n", i, node->outputs->data[0]);
      invoke_status = registration->invoke(&context_, node);
      printf("}\n");
      // Cleanup
      context_.tensors[node->inputs->data[0]].type &= 0xFF;

      if (invoke_status == kTfLiteError) {
        TF_LITE_REPORT_ERROR(
            error_reporter_,
            "Node %s (number %d) failed to invoke with status %d",
            OpNameFromRegistration(registration), i, invoke_status);
        return kTfLiteError;
      } else if (invoke_status != kTfLiteOk) {
        return invoke_status;
      }
    }
  }
  return kTfLiteOk;
}

TfLiteTensor* MicroInterpreter::input(size_t index) {
  const size_t length = inputs_size();
  if ((index < 0) || (index >= length)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Input index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return &(context_.tensors[inputs().Get(index)]);
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const size_t length = outputs_size();
  if ((index < 0) || (index >= length)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Output index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return &(context_.tensors[outputs().Get(index)]);
}

TfLiteTensor* MicroInterpreter::tensor(size_t index) {
  const size_t length = tensors_size();
  if ((index < 0) || (index >= length)) {
    TF_LITE_REPORT_ERROR(error_reporter_,
                         "Tensor index %d out of range (length is %d)", index,
                         length);
    return nullptr;
  }
  return &context_.tensors[index];
}

TfLiteStatus MicroInterpreter::ResetVariableTensors() {
  const size_t length = tensors_size();
  for (size_t i = 0; i < length; ++i) {
    TfLiteTensor* cur_tensor = tensor(i);
    if (cur_tensor->is_variable) {
      TfLiteStatus status = tflite::ResetVariableTensor(cur_tensor);
      if (status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter_,
                             "Failed to reset variable tensor at index: %d", i);
        return status;
      }
    }
  }
  return kTfLiteOk;
}

}  // namespace tflite
