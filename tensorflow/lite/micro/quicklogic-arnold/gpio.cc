
// #if defined(__riscv__)
	// #error "RV"
// #endif

#include <stdio.h>
//#include "fc_config.h"
// #if defined(__riscv__)
	// #error "RV"
// #endif
//#include "hal/pulp.h"
#include "gpio.h"

#define APB_SOC_CTRL_BASE    0x1A104000
#define APB_SOC_REG_PADFUN0  (APB_SOC_CTRL_BASE + 0x10)
#define APB_SOC_FPGA_CLK_SEL (APB_SOC_CTRL_BASE + 0xE0)
#define APB_SOC_FPGA_RESET   (APB_SOC_CTRL_BASE + 0xE8)
#define APB_SOC_FPGA_CLKGATE (APB_SOC_CTRL_BASE + 0xE4)
#define APB_SOC_FPGA_GATE    (APB_SOC_CTRL_BASE + 0xEC)

#define GPIO_START_ADDR   0x1A101000
#define GPIO_PADCFG       (GPIO_START_ADDR + 0x28)
#define GPIO5_OUT         (GPIO_START_ADDR + 0x0C)

void padcfg(void)
{
  volatile unsigned int* apb_func0        = (unsigned int *)APB_SOC_REG_PADFUN0;
  volatile unsigned int* apb_gpio_cfg     = (unsigned int *)GPIO_PADCFG;
  volatile unsigned int* apb_gpio_dir     = (unsigned int *)GPIO_START_ADDR;
	volatile unsigned int* gpio5_cs1        = (unsigned int *)GPIO5_OUT;


  *(gpio5_cs1)    = 0;
  *(apb_func0)    = 0x0400; //01_00-00_00-00_00 means PAD 0,1,2,3 and 4 come from peripherals and PAD 5 comes from SW GPIO (CS1)
  *(apb_gpio_cfg) = 0x202222; //xx10_xxxx_xx10_xx10_xx10_xx10 means PAD 0,1,2,3 and 5 have PD not enable and PU enable
  *(apb_gpio_dir) = 0x20; // 100000 means PAD 5 direction, i.e. Output Enable = 1
}

void setgpio5(int value) {
	  volatile unsigned int* gpio5_cs1        = (unsigned int *)GPIO5_OUT;
		
    *(gpio5_cs1)    = value << 5; //write value to PAD5
}