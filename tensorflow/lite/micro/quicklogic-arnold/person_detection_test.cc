/*============================================================================

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

#define PULP_CHIP_STR arnold
#include "fc_config.h"
#define ARCHI_EU_ADDR 0x00020800
#define ARCHI_EU_OFFSET 0x00000800
#define ARCHI_DEMUX_ADDR 0x00024000
#include "fll.h"
#include "gpio.h"
extern "C" {
#include "hal/timer/timer_v2.h"
#include "hal/eu/eu_v1.h"
#include "pulp.h"
int i2c_16read16 (char dev, int addr) ;  
}

#include "arnold_apb_ctl.h"
#include "apb_conv2d.h"
#define USE_UART
#undef USE_UART   // remove this to use the uart to program the fpga
#ifdef USE_UART
#include "programFPGA.h"
unsigned int __rt_iodev = 1;
unsigned int __rt_iodev_uart_baudrate = 460800;
unsigned int __rt_iodev_uart_channel = 0;
static rt_camera_t *camera;
#endif

bool fpga_programmed;
bool camera_present;
// Create an area of memory to use for input, output, and intermediate arrays.
constexpr int tensor_arena_size = 93 * 1024;
uint8_t tensor_arena[tensor_arena_size] __attribute__ ((aligned (16)));

extern rt_camera_t *setup_camera();


TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestInvoke) {
  // Set up logging.
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  apb->padfunc0 = 0x000002aa;  // UART(7,8), gpio(5,6), fpga (0-4), camera(9-19)
  gpio->dir31_00 = gpio->dir31_00 | (7 << 4);  // gpio 5,6,7 output
  gpio->enable31_00 = gpio->enable31_00 | (7 << 4); // gpio output enable
  gpio->out31_00 = 0; // clear all gpio

  prog_fll(0,20000,2);  // 326 MHz  goto 456 in 2 steps/
  prog_fll(0,28000,2);  // 456 MHz
  prog_fll(2,2950,3);  // 25 MHz -- 
  prog_fll(2,6100,3);  // 50 MHz

  apb->fpga_clk = 2;
  fpga_programmed = false;
  camera_present = false;
#ifdef USE_UART
  rt_uart_conf_t conf;
  rt_event_alloc(NULL,4);
  rt_uart_conf_init(&conf);
  conf.itf = 0;
  conf.baudrate = 460800;
  rt_uart_t *uart = rt_uart_open (NULL,&conf, NULL);
  if (uart== NULL) {
    printf("Failed to open uart\n");
    exit(0);
  }
  programFPGA(uart,"tfl");
  fpga_programmed = true;
  apb->fpga_reset = 1;
  apb->fpga_reset = 0xF;
  apb->fpga_gate  = 0xFFFF;
  efpga->i2c = 3;  // reset i2c bits
  int chipid;

  chipid = i2c_16read16((char)0x24,(int)0);
  printf("Camera id = %04x\n",chipid);
  if (chipid == 0x01b0)  // 0x1b0 == Himax sensor
    camera_present = true;
  if (camera_present) {
    printf("calling setup_camera\n");
    camera = setup_camera();
    printf("calling setup_camera\n");
  }
#endif
  printf("Arnold GPIO 5,6= software, GPIO 0 = hardware\n");

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
  // SHow us the picture
#ifdef USE_UART
  printf("ScReEn96 96\n");
  for (int j = 0; j<96; j++) {
      for (int k = 0; k < 96; k += 32) {
	int l = 0;
	printf("ImAgE %x %x",j,k);
	while (l < 32) 
	  printf(" %02x",person_data[j*96+k+(l++)]);
	printf("\n");
	for (int m=0; m < 20000; m++) asm volatile("nop");
      }
    }
#endif
    
  
  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = person_data[i];
  }

  // Run the model on this input and make sure it succeeds.
  gpio->out31_00 = (1<<6);
  TfLiteStatus invoke_status = interpreter.Invoke();
  gpio->out31_00 = 0;
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
#ifdef USE_UART
  for (int j = 0; j<96; j++) {
      for (int k = 0; k < 96; k += 32) {
	int l = 0;
	printf("ImAgE %x %x",j,k);
	while (l < 32) 
	  printf(" %02x",no_person_data[j*96+k+(l++)]);
	printf("\n");
	for (int m=0; m < 20000; m++) asm volatile("nop");
      }
    }
#endif

  for (int i = 0; i < input->bytes; ++i) {
    input->data.uint8[i] = no_person_data[i];
  }

  // Run the model on this "No Person" input
  gpio->out31_00 = (1<<6);
  invoke_status = interpreter.Invoke();
  gpio->out31_00 = 0;
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


  #ifdef USE_UART

  int display_on = 0;
  char rx_buffer[20];   
  rx_buffer[0] = 1;
  while(camera_present) {
    rt_event_t *event1 = rt_event_get_blocking(NULL);
    rt_camera_capture (camera,tensor_arena, 324*244, event1);
    rt_cam_control(camera,CMD_START, 0);
    rt_event_wait(event1);
    rt_cam_control(camera,CMD_STOP, 0);
        if (rx_buffer[0] == 'V') {
      display_on = 2;
    }
    if (rx_buffer[0] == 'C') {
      display_on = 1;
    }
    if (rx_buffer[0] == 'c') {
      display_on = -1;
    }
    else if (rx_buffer[0] == 'Q') {
      printf("Exiting\n");
      break;
    }
    if (rx_buffer[0] != 0) {
      rx_buffer[0] = 0;
      rt_event_t *event = rt_event_get_blocking(NULL);
      rt_uart_read(uart, rx_buffer,1,event);
    }

    if (display_on == 2) {
      printf("ScReEn320 320\n");
      for (int j = 0; j<240; j++) {
	for (int k = 0; k < 320; k += 32) {
	  int l = 0;
	  printf("ImAgE %x %x",j,k);
	  while (l < 32) 
	    printf(" %02x", tensor_arena[650+j*324+k+(l++)]);
	  printf("\n");
	  for (int m=0; m < 20000; m++) asm volatile("nop");
	}
      }
      display_on = 0;
    }

    for (int i = 0; i < 96; ++i) {
      for (int j = 0; j<96; j++)
	input->data.uint8[i*96+j] = tensor_arena[74*324+i*324+144+j];
  }

  if (display_on) {
    if (display_on > 0)
      printf("ScReEn96 96\n");
    for (int j = 0; j<96; j++) {
      for (int k = 0; k < 96; k += 32) {
	int l = 0;
	printf("ImAgE %x %x",j,k);
	while (l < 32) 
	  printf(" %02x",input->data.uint8[j*96+k+(l++)]);
	printf("\n");
	for (int m=0; m < 20000; m++) asm volatile("nop");
      }
    }
    display_on--;
  }

  // Run the model on this "No Person" input
  gpio->out31_00 = (1<<6);
  invoke_status = interpreter.Invoke();
  gpio->out31_00 = 0;
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
      "Himax Camera image.  person score: %d, no person score: %d\n", person_score,
      no_person_score);
  //  TF_LITE_MICRO_EXPECT_GT(no_person_score, person_score);

  //  TF_LITE_REPORT_ERROR(error_reporter, "Ran successfully\n");
  
  }
#endif  
}

TF_LITE_MICRO_TESTS_END
