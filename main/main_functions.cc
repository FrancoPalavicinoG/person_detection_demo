#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
#include "driver/gpio.h"

#define LED_RED_GPIO static_cast<gpio_num_t>(12)
#define LED_GREEN_GPIO static_cast<gpio_num_t>(13)
#define BUTTON_GPIO static_cast<gpio_num_t>(15)
#define FLASH_GPIO static_cast<gpio_num_t>(4)

bool button_pressed = false;
bool alarm_active = false;  // Inicia con la alarma desactivada

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;

#ifdef CONFIG_IDF_TARGET_ESP32S3
constexpr int scratchBufSize = 40 * 1024;
#else
constexpr int scratchBufSize = 0;
#endif
constexpr int kTensorArenaSize = 81 * 1024 + 9000;
static uint8_t *tensor_arena;
}  // namespace

void setup() {
  // Todos mis pines estan negados por eso todo esta invertido
  gpio_reset_pin(LED_GREEN_GPIO);
  gpio_set_direction(LED_GREEN_GPIO, GPIO_MODE_OUTPUT);

  gpio_reset_pin(LED_RED_GPIO);
  gpio_set_direction(LED_RED_GPIO, GPIO_MODE_OUTPUT);

  gpio_reset_pin(FLASH_GPIO);
  gpio_set_direction(FLASH_GPIO, GPIO_MODE_OUTPUT);

  gpio_reset_pin(BUTTON_GPIO);
  gpio_set_direction(BUTTON_GPIO, GPIO_MODE_INPUT);

  gpio_set_level(LED_RED_GPIO, 1);  
  gpio_set_level(LED_GREEN_GPIO, 0);
  gpio_set_level(FLASH_GPIO, 0);

  model = tflite::GetModel(g_person_detect_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  if (tensor_arena == NULL) {
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  static tflite::MicroMutableOpResolver<10> micro_op_resolver;
   micro_op_resolver.AddAveragePool2D();
   micro_op_resolver.AddConv2D();
   micro_op_resolver.AddDepthwiseConv2D();
   micro_op_resolver.AddReshape();
   micro_op_resolver.AddSoftmax();
   micro_op_resolver.AddMaxPool2D(); 
   micro_op_resolver.AddFullyConnected(); 
   micro_op_resolver.AddLogistic(); 
   micro_op_resolver.AddQuantize();
   micro_op_resolver.AddDequantize();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);

#ifndef CLI_ONLY_INFERENCE
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

void loop() {
  if (gpio_get_level(BUTTON_GPIO) == 0 && !button_pressed) {
    printf("Boton presionado\n");
    button_pressed = true;
    alarm_active = !alarm_active; 

    if (alarm_active) {
      gpio_set_level(LED_RED_GPIO, 0);
      gpio_set_level(LED_GREEN_GPIO, 1);
    } else {
      gpio_set_level(LED_RED_GPIO, 1);
      gpio_set_level(LED_GREEN_GPIO, 0);
    }

    vTaskDelay(200 / portTICK_PERIOD_MS);  // Debounce delay
  } else if (gpio_get_level(BUTTON_GPIO) == 1) {
    printf("Boton No presionado\n");
    button_pressed = false;
  }

  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);

  // Process the inference results.
  int8_t person_score = output->data.uint8[kPersonIndex];
  int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  float person_score_f =
      (person_score - output->params.zero_point) * output->params.scale;
  float no_person_score_f =
      (no_person_score - output->params.zero_point) * output->params.scale;

  // Respond to detection
  RespondToDetection(person_score_f, no_person_score_f);
  vTaskDelay(1);  // to avoid watchdog trigger
}

void app_main() {
  setup();

  // Iniciar el bucle principal
  while (true) {
    loop();
  }
}
