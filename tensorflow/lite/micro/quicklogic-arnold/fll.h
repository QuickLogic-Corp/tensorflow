#define REFCLK 32768
typedef union {
  struct {
    unsigned int act_mult:16;  /* Fll current multiplication factor */
    unsigned int reserved:16;
  };
  unsigned int raw;
} fll_reg_stat_t;

typedef union {
  struct {
    unsigned int mult_factor:16;      /* Fll requested multiplication factor, reset: 0x5f5.
                                         If RefClk=32768 and Div=2 Freq= 24.98 MHz */
    unsigned int dco_input:10;        /* DCO input code for stand alone mode, reset: 0x121 */
    unsigned int clock_out_divider:4; /* Fll clock output divider, reset: 0x1 e.g div 2 */
    unsigned int output_lock_enable:1;/* Fll output gated by lock signal (active high), reset 1 */
    unsigned int mode:1;              /* Fll mode. 0: stand alone (unlocked), 1: normal, reset 0 */
  };
  unsigned int raw;
} fll_reg_cfg1_t;

typedef union {
  struct {
    unsigned int loop_gain:4;         /* Fll loop gain, reset: 0x9 */
    unsigned int de_assert_cycles:6;  /* Normal mode: number of refclock unstable cycles till lock de-assert
                                         Standalone mode: lower 6 bits of lock assert counter. Reset: 0x10 */
    unsigned int assert_cycles:6;     /* Normal mode: number of refclock stable cycles till lock assert
                                         Standalone mode: upper 6 bits of lock assert counter. Reset: 0x10 */
    unsigned int lock_tolerance:12;   /* Lock tolerance: margin arounf the target mult factor where clock is
                                         considered as stable. Reset: 0x200
                                         With Fmax=250Mhw (Div=2^4), Fmin=32K (Div=2^15)
                                         Tolerance: 32K*(512/16)=1.048MHz .. 512 Hz */
    unsigned int pad:1;
    unsigned int config_clock_sel:1;  /* Select ref clock when mode = standalone, 0:RefClock, 1: DcoOut. Reset:1 */
    unsigned int open_loop:1;         /* When 1 clock operates in open loop when locked */
    unsigned int dithering:1;         /* When 1 Dithering is enabled */
  };
  unsigned int raw;
} fll_reg_cfg2_t;

typedef union {
  struct {
    unsigned int pad1:6;
    unsigned int state_fract_part:10; /* Integrator state: fractional part (dithering input unit) */
    unsigned int state_int_part:10;   /* Integratot state: integer part (DCO input bits) */
    unsigned int pad2:6;
  };
  unsigned int raw;
} fll_reg_integ_t;


typedef struct {
  fll_reg_stat_t stat;
  fll_reg_cfg1_t cfg1;
  fll_reg_cfg2_t cfg2;
  fll_reg_integ_t integ;
} fll_t ;


int spow(int);
int dump_fll(int);
int prog_fll(int fll, int mult, int div);
