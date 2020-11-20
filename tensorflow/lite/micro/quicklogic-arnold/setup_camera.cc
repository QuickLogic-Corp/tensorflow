#define PULP_CHIP_STR arnold
#include "fc_config.h"
#define ARCHI_EU_ADDR 0x00020800
#define ARCHI_EU_OFFSET 0x00000800
#define ARCHI_DEMUX_ADDR 0x00024000

extern "C" {
#include <rt/rt_api.h>
}

rt_camera_t *setup_camera() {

  rt_camera_t *camera;
  rt_cam_conf_t cam_conf;
  if (rt_event_alloc(NULL, 8)) return 0;
  rt_camera_conf_init(&cam_conf);
  cam_conf.id = 0;
  cam_conf.type = RT_CAM_TYPE_HIMAX;
  cam_conf.control_id = 1;
  printf("Calling camera_open\n");
  camera = rt_camera_open(0, &cam_conf, 0);
  if (camera == 0) return 0;
  printf("Calling cam control\n");
  rt_cam_control(camera, CMD_INIT, 0);

  return (camera);
}
  
  
