/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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


#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

#include "accel.h"

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"



namespace tflite {

//namespace reference_ops {


void ConvAccelSIMD(const ConvParams& params, 
                  const RuntimeShape& input_shape, const uint8* input_data, 
                  const RuntimeShape& filter_shape, const int8_t* filter_data, 
                  const RuntimeShape& bias_shape, const int32* bias_data, 
                  const RuntimeShape& output_shape, uint8* output_data, 
                  const RuntimeShape& im2col_shape, uint8* im2col_data, 
                  void* cpu_backend_context,
                  bool fPrint) {
  printf("ConvAccelSIMD\n");
  
  int32_t iprintcol;
  if (fPrint) {
    iprintcol = 0;
    printf("int32_t axOutput[] = {\n  ");
  }
 
  (void)cpu_backend_context;  // only used in optimized code.
  (void)im2col_data;   // only used in optimized code.
  (void)im2col_shape;  // only used in optimized code.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  
  int first = 0;
  // int32 filter_val_max = -10000;
  // int32 filter_val_min =  10000;
  // int32 filter_act_max = -10000;
  // int32 filter_act_min =  10000;
  // int32 accmin =  1000000000;
  // int32 accmax = -1000000000;
  // int32 outmin_orig =  1000000000;
  // int32 outmax_orig = -1000000000;
  // int32 outmin_new =  1000000000;
  // int32 outmax_new = -1000000000;
  // int32 diff, diff_max = -1000000, diff_min = 1000000;
  int32 acc_orig;
  int32 acc_new;
  // int32 shift;
  // int32 quant_mask = 0xFC000000;
  // int32 output_multiplier_quant;
  // shift = (output_multiplier < 0x60000000) ? (-output_shift): (-output_shift);
  // output_multiplier_quant = output_multiplier & quant_mask;
  // int icol = 0;
  const int simd = {8};
  printf("batches=%d\n", batches);
  printf("output_depth=%d\n", output_depth);
  
  for (int batch = 0; batch < batches; ++batch) {
    
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
            for (int isimd = 0; isimd != simd; isimd++) {
              const int in_x_origin = (out_x * stride_width) - pad_width;
              const int in_y_origin = (out_y * stride_height) - pad_height;
              int32_t acc = 0;
              for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                  for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                    const int in_x = in_x_origin + dilation_width_factor * filter_x;
                    const int in_y =
                        in_y_origin + dilation_height_factor * filter_y;
                    // If the location is outside the bounds of the input image,
                    // use zero as a default value.
                    if ((in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                        (in_y < input_height)) {
                      int32_t ifilt = (((out_channel * input_depth) + in_channel) * simd) + isimd;
                      int32_t filter_val = filter_data[ifilt];
                      int32_t input_val = input_data[Offset(input_shape, batch, in_y,
                                                          in_x, in_channel)];
                                                          
                      if (first++ < 100) {
                        //printf("%d,%d,0x%02x,0x%02x\n", first, ifilt, filter_val, input_val);
                      }
                      
                      // ***Expect this to be done in the accelerator*** //
                      input_val = input_val - 128;
                      
                      

                      // filter_val_max = std::max(filter_val_max, filter_val);
                      // filter_val_min = std::min(filter_val_min, filter_val);

                      
                      // filter_act_max = std::max(filter_act_max, filter_val);
                      // filter_act_min = std::min(filter_act_min, filter_val);

                      acc += filter_val * input_val;
                      //printf("fv, iv, acc: %02x, %02x, %02x\n", filter_val, input_val, acc);
                    }
                  }
                }
              }
              if (bias_data) {
                acc += bias_data[out_channel * simd + isimd];  // Includes output_offset
                                                // Scaled so we can move it prior to quant
                                                // By bringing it earlier, quantization is now linear, not affine
                                                // NOTE: did add this into bias_data
              }
              //printf("acc+bias: %d\n", acc);
              // if (fPrintOut) {
                // printf(" 0x%08x,", acc);
                // icol++;
                // if (icol == 8) {
                  // icol = 0;
                  // printf("\n");
                // }
              // }
              // acc += output_offset; // Scaled so we can move it prior to quant
                                    // // By bringing it earlier, quantization is now linear, not affine
                                    // // NOTE: should just add this into bias_data
              // accmax = std::max(accmax, acc);
              // accmin = std::min(accmin, acc);
              acc_orig = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
              // acc_orig = MultiplyByQuantizedMultiplier(acc, 0x7fffffff, output_shift);
              
              int shift = -output_shift;
              int p1 = (acc < 0) ? 0 : (acc >> shift);
              int p2 = (((p1 + 2) >> 2) & 0x7F);
              int p3 = ((output_multiplier + (1<<23)) >> 24) & 0x7F;
              int p4 = p2 * p3;
              int p5 = ((p4 + (1 << 5)) >> 5);
              // acc_new = p5;
              
              int q1 = (acc < 0) ? 0 : (acc >> (shift-1));
              int q2 = ((q1 + 2) >> 2);
              int q2p = (q2>255) ? 255 :(q2 & 0xFF);
              int q3 = ((output_multiplier + (1<<23)) >> 24) & 0xFF;
              int q4 = q2p * q3;
              int q5 = ((q4 + (1 << 5)) >> 6);
              acc_new = q5;
              
              int r1 = ((output_multiplier + (1<<23)) >> 24) & 0xFF;
              int r2 = (1 << 14)/r1;
              int r3 = (acc < 0) ? 0 : ((acc + (1 << shift)) >> (shift + 1));
              int r4 = (r3 > r2) ? (r1 * r2) : ((r1 & 0xF0) * ((r3+0) & 0xFF));
              int r5 = ((r4 + (1 << 5)) >> 6);
              acc_new = r5;
              
              int sc1 = ((output_multiplier + (1<<23)) >> 24) & 0xFF;
              int sc2 = (1 << 14)/sc1;
              int sc3 = sc1 * sc2;
              int sv4 = (acc < 0) ? 0 : ((acc + (1 << shift)) >> (shift + 1));
              int sv5 = (sv4 > sc2) ? (sc3) : (sc1 * (sv4 & 0xFF));
              int sv6 = ((sv5 + (1 << 5)) >> 6);
              acc_new = sv6;
              
              int tc1 = 0xFF & ((output_multiplier + (1<<23)) >> 24);                 // Load time: (uint8_t)
              int tc2 = (1 << (14 + shift + 1))/tc1;                                  // Load time 
              // int tc1 = output_multiplier;
              // int tc2 = output_offset;
              int tv5 = tc1 * (0xFF & (acc + (1 << shift)) >> (shift + 1));
              int tv6 = (acc < 0) ? 0 : ((acc < tc2) ? ((tv5 + (1 << 5)) >> 6) : 255);
              acc_new = tv6;
              
              // int32_t uc1 = output_multiplier / (1<<16);                // Load time
              // int16_t uc2 = uc1;                                        // Load time 
              // uc2 = output_multiplier;
              
              // int16_t uv1 = acc >> 7;
              // int32_t uv2 = uc2 * uv1;
              // int16_t uv3 = uv2 >> (shift + 7);
              // acc_new = uv3;
              
              //int16_t uv1 = acc >> (shift - 5);
              // int16_t uv1 = acc >> shift;
              int32_t uv2 = (int16_t)output_multiplier * (int16_t)(acc >> shift);
              int16_t uv3 = uv2 >> 16;
              int16_t uv4 = uv3 >> 4;
              acc_new = uv4;
              
              
              // if (first < 10) {
                // printf("output_multiplier=%d, shift=%d, uc1=%d, uc2=%d\n", output_multiplier, shift, uc1, uc2);
                // printf("acc=%d, uv1=%d, uv2=%d, uv3=%d, uv4=%d acc_new=%d, acc_orig=%d\n", acc, uv1, uv2, uv3, uv4, acc_new, acc_orig);
                // first++;
              // }
              
              // diff = (acc_new >= 0 && acc_orig >= 0) ? (acc_new - acc_orig) : 0;
              // diff_max = std::max(diff_max, diff);
              // diff_min = std::min(diff_min, diff);
              
              // if (diff < -100 || diff == 385) {
                // printf("diff = %d\n", diff);
                // printf("shift, omult, acc, r1, r2, r2p, r3, r4, r5\n");
                // printf("%x, %x, %x, %x, %x, %x, %x, %x, %x\n", shift, output_multiplier, acc, r1, r2, r2p, r3, r4, r5);
                // printf("%d, %d, %d, %d, %d, %d, %d, %d, %d\n", shift, output_multiplier, acc, r1, r2, r2p, r3, r4, r5);
                // printf("orig:new = %d:%d\n", acc_orig, acc_new);
              // }
              
              // outmax_orig = std::max(outmax_orig, acc_orig);
              // outmin_orig = std::min(outmin_orig, acc_orig);
              // outmax_new = std::max(outmax_new, acc_new);
              // outmin_new = std::min(outmin_new, acc_new);
              
              acc_orig = std::max(acc_orig, output_activation_min);
              acc_orig = std::min(acc_orig, output_activation_max);
              acc_new = std::max(acc_new, output_activation_min);
              acc_new = std::min(acc_new, output_activation_max);
              
              // diff = acc_new - acc_orig;
              // diff_max = std::max(diff_max, diff);
              // diff_min = std::min(diff_min, diff);
              
              if (true) {
                output_data[Offset(output_shape, batch, out_y, out_x, out_channel * simd + isimd)] = static_cast<uint8>(acc_new);
                //printf("Out(%d,%d,%d,%d)=%d\n", batch, out_y, out_x, out_channel * simd + isimd, acc_new);
              } else {
                output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<uint8>(acc_orig);
              }
              if (fPrint) {
                printf("0x%08x, %d%s", acc, acc_new, (iprintcol == 7) ? ",\n  " : ", ");
                iprintcol++;
                if (iprintcol == 8) iprintcol = 0;
              }
            }
          }
        }
      }
   
  }
  if (fPrint) {
    printf("\n};\n");
  }
  // printf("filter_val: [%d, %d]\n", filter_val_min, filter_val_max);
  // printf("filter_act: [%d, %d]\n", filter_act_min, filter_act_max);  
  // printf("raw_acc:    [%d, %d]\n", accmin, accmax);
  // printf("shift:       %d\n", shift);
  // printf("input_offset:  %d\n", input_offset);
  // printf("out_val_orig: [%d, %d]\n", outmin_orig, outmax_orig);
  // printf("out_val_new:  [%d, %d]\n", outmin_new, outmax_new);
  // printf("quant_mask:   0x%08x\n", quant_mask);
  // printf("diff:  [%d, %d]\n", diff_min, diff_max);
}


//}  // namespace reference_ops
}  // namespace tflite

