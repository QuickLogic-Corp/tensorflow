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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/examples/person_detection/model_settings.h"
#include "tensorflow/lite/micro/examples/person_detection/no_person_image_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_detect_model_data.h"
#include "tensorflow/lite/micro/examples/person_detection/person_image_data.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_optional_debug_tools.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "fll.h"
#include "gpio.h"
//#include "programFPGA.h"
#include "arnold_apb_ctl.h"
#include "apb_conv2d.h"
/*
unsigned int __rt_iodev = 1;
unsigned int __rt_iodev_uart_baudrate = 115200;
unsigned int __rt_iodev_uart_channel = 0;
*/
// Create an area of memory to use for input, output, and intermediate arrays.
constexpr int tensor_arena_size = 93 * 1024;
uint8_t tensor_arena[tensor_arena_size] __attribute__ ((aligned (16)));

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  //  printf("Attempted to program FLL0 %d Hz\n",prog_fll(0,28000,2));  // 456 MHz
  //  printf("Attempted to program FLL1 %d Hz\n",prog_fll(1,6100,3));   //50 MHz
  //  printf("Attempted to program FLL2 %d Hz\n",prog_fll(2,7400,3));  // 61 MHs
  prog_fll(0,20000,2);  // 326 MHz  got 456 in 2 steps/
  prog_fll(0,28000,2);  // 456 MHz
  // prog_fll(1,6100,2);  // 50 MHz
  prog_fll(2,6100,3);  // 50 MHz 
  
  apb->fpga_clk = 2;
  //programFPGA();
  apb->fpga_reset = 1;
  apb->fpga_reset = 0xF;
  apb->fpga_gate  = 0xFFFF;
  printf("Arnold test bed with gpio\n");
  int fref = 6;
  int fout = 480;

  padcfg();
  /*  setgpio5(0);
  prog_fll(0, fref, fout/fref,1);
  setgpio5(0);
  prog_fll(0, fref, fout/fref,1);
  setgpio5(0);
  prog_fll(0, fref, fout/fref,1);
  setgpio5(1);
  prog_fll(0, fref, fout/fref,1);
  setgpio5(0);
  prog_fll(0, fref, fout/fref,1);
  */

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.\n",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }
  //PrintModelData(model, error_reporter);

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  tflite::MicroMutableOpResolver<3> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();

  // Build an interpreter to run the model with.
  tflite::MicroInterpreter interpreter(model, micro_op_resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  interpreter.AllocateTensors();

  // Get information about the memory area to use for the model's input.
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect.
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  TF_LITE_MICRO_EXPECT_EQ(4, input->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, input->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(kNumRows, input->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(kNumCols, input->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kNumChannels, input->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, input->type);

  // Copy an image with a person into the memory area used for the input.
  const uint8_t* person_data = g_person_data;
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = person_data[i];
  }

  // Run the model on this input and make sure it succeeds.
  setgpio5(1);
  TfLiteStatus invoke_status = interpreter.Invoke();
  setgpio5(0);
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // Make sure that the expected "Person" score is higher than the other class.
  uint8_t person_score = output->data.uint8[kPersonIndex];
  uint8_t no_person_score = output->data.uint8[kNotAPersonIndex];
  TF_LITE_REPORT_ERROR(error_reporter,
                       "person data.  person score: %d, no person score: %d\n",
                       person_score, no_person_score);
  TF_LITE_MICRO_EXPECT_GT(person_score, no_person_score);

  // Now test with a different input, from an image without a person.
  const uint8_t* no_person_data = g_no_person_data;
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = no_person_data[i];
  }

  // Run the model on this "No Person" input.
  setgpio5(1);
  invoke_status = interpreter.Invoke();
  setgpio5(0);
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed\n");
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Get the output from the model, and make sure it's the expected size and
  // type.
  output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(4, output->dims->size);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[0]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[1]);
  TF_LITE_MICRO_EXPECT_EQ(1, output->dims->data[2]);
  TF_LITE_MICRO_EXPECT_EQ(kCategoryCount, output->dims->data[3]);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteUInt8, output->type);

  // Make sure that the expected "No Person" score is higher.
  person_score = output->data.uint8[kPersonIndex];
  no_person_score = output->data.uint8[kNotAPersonIndex];
  TF_LITE_REPORT_ERROR(
      error_reporter,
      "no person data.  person score: %d, no person score: %d\n", person_score,
      no_person_score);
  TF_LITE_MICRO_EXPECT_GT(no_person_score, person_score);

  TF_LITE_REPORT_ERROR(error_reporter, "Ran successfully\n");
}

TF_LITE_MICRO_TESTS_END
