
#define PULP_CHIP_STR arnold
#include "fc_config.h"
#define ARCHI_EU_ADDR 0x00020800
#define ARCHI_EU_OFFSET 0x00000800
#define ARCHI_DEMUX_ADDR 0x00024000
extern "C" {
#include "hal/timer/timer_v2.h"
#include "hal/eu/eu_v1.h"
#include "pulp.h"
}
#include "programFPGA.h"
#include "apb_fcb.h"
//#include "fourbyte.h"
void programFPGA()
{

    int bit_line_count  = 172; // 0x00ac
    int word_line_count = 336; // 0x0150
    int tot_word_cnt = bit_line_count * word_line_count;
    int tot_byte_cnt = bit_line_count * word_line_count * 4; //0x038700
    int i;
    // ArcticPro(TM) 2 eFPGA configuration setup programming
    FCBAPB_BL_PW_CFG_0 =0xff; // required
    FCBAPB_BL_PW_CFG_1 =0xff; // required
    FCBAPB_WL_PW_CFG =0xff; // required
    FCBAPB_CFG_MIS_0 =0x30; // [7:6] - System Clock Frequency - 10 MHz
    //[5] VLP_PIN_EN – disabled
    FCBAPB_WRD_CNT_B0 =0x00; // tot_byte_count - bottom nibble
    FCBAPB_WRD_CNT_B1 =0x87; // tot_byte_count - middle nibble
    FCBAPB_WRD_CNT_B2 =0x03; // tot_byte_count - top nibble
    FCBAPB_BL_CNT_L =0xac; // bit_line_count - low nibble

    FCBAPB_BL_CNT_H =0x00; // bit_line_count - high nibble
    FCBAPB_WL_CNT_L =0x50; // word_line_count - low nibble
    FCBAPB_WL_CNT_H =0x01; // word_line_count - high nibble
    FCBAPB_CFG_WRP_CCNT =0x06; // Cfg Programming Signal Pulse Width
    FCBAPB_SCRATCH_BYTE =0x00; // [7:6] CFG_HOLD
    FCBAPB_FB_CFG_CMD =0x00; // 0x00 - Configuration Write
    // 0x01 - Configuration Write w/ Pre-Checksum
    // 0x02 - Configuration Write w/ Post-Checksum
    // 0x10 - Quad SLC Configuration
    // 0x11 - Quad SLC Configuration w/Pre-Check
    // 0x40 - Configuration READ
    // Always clear FB_CFG_DONE before starting configuration

    if ( FCBAPB_FB_CFG_DONE == 1 ) {
        FCBAPB_FB_CFG_DONE = 0;
    }

    // kick-off command for ArcticPro 2 eFPGA configuration
    // Must do last after setup
    FCBAPB_FB_CFG_KICKOFF = 0x1;


    // wait 20 us for internal reset to ArcticPro 2 to clear eFPGA
    for(i = 0; i<2800; i++) asm volatile("nop");


    {
      extern int __rt_nb_devices;
      char rx_buffer[] = "LoAd tfl \n      ";
      rt_event_alloc(NULL,4);
      rt_uart_conf_t conf;
      rt_uart_conf_init(&conf);
      conf.itf = 0;
      conf.baudrate = 115200;
      rt_uart_t *uart = rt_uart_open (NULL,&conf, NULL);
      if (uart== NULL) {
            printf("Failed to open uart\n");
	    exit(0);
      } else {
      rt_event_t *event = rt_event_get_blocking(NULL);
      rt_uart_write(uart,rx_buffer, sizeof(rx_buffer), NULL);
      int count = 0;
      while (count < tot_word_cnt) {
	  rt_event_t *event = rt_event_get_blocking(NULL);
	rt_uart_read(uart, rx_buffer,4, event);
	rt_event_wait(event);
	FCBAPB_RW_DATA_PORT = rx_buffer[0] | (rx_buffer[1] << 8) |
		  (rx_buffer[2] << 16) | (rx_buffer[3] << 24);
	count++;
      }
      }
    }

    
    // WRITE in the configuration data into ArcticPro 2 eFPGA through Configuration Controller for APB
    //for ( i=0; i < tot_word_cnt; i++ ) {
    //  FCBAPB_RW_DATA_PORT = 1; //fb_cfg_data[i]; // 32-bit register
    //}
    FCBAPB_FB_CFG_DONE = 1; // in APB mode, this needs to be explicitly written at end of configuration


    // wait 5 us just in case
    for(i = 0; i<2800/4; i++) asm volatile("nop");

}
