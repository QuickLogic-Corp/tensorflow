#include "fll.h"
#include <stdio.h>

int spow2 (int exp) {
  int i, ret;
  ret = 1;
  if (exp == 0) return ret;
  else
    for (i = 0; i < exp; ret = ret * 2, i++);
  return ret;
}

int prog_fll(int f, int mult, int div) {
  fll_t *fll = (fll_t*)0x1a100000;
  int mult_out;
  int ret = (REFCLK * (mult+1))/spow2(div?div-1:0);
  fll_reg_cfg1_t i;
  fll_reg_cfg2_t j;
  fll_reg_integ_t k;
  i.mult_factor = mult;
  i.dco_input = 0xf0;
  i.clock_out_divider = div;
  i.output_lock_enable = 0;
  i.mode = 1;

  //  printf("cfg1 mult=%04x, dco_inp=%03x, div=%d, locken=%d, mode=%d\n",
  //	 i.mult_factor, i.dco_input, i.clock_out_divider, i.output_lock_enable,
  //	 i.mode);

  j.loop_gain = 7;
  j.de_assert_cycles = 6;
  j.assert_cycles = 0x10;
  j.lock_tolerance = 0x50;
  j.config_clock_sel = 1;
  j.open_loop=0;
  j.dithering=1;
//  printf("cfg2 loop gain=%x de_assert=%2x assert=%2x tolerance=%3x clk_sel=%x open_loop=%x,ditering=%x\n",
//       j.loop_gain, j.de_assert_cycles,
//	 j.assert_cycles, j.lock_tolerance, j.config_clock_sel,
//	 j.open_loop, j.dithering);
  fll[f].cfg2.raw = j.raw;
  fll[f].cfg1.raw = i.raw;
  k.raw = fll[f].integ.raw;
// printf("FLL%d m=%d (%04x), div= %d %04x %08x (0x%x.0x%x)\n",f,
//	 i.mult_factor,
//	 i.mult_factor,
//	 i.clock_out_divider,
//	 fll[f].stat.act_mult,
//	 fll[f].cfg1.raw,
//	 fll[f].cfg2.raw,
//	 k.state_int_part,k.state_fract_part
//	 );
  ret  = (REFCLK * (mult_out+1))/spow2(div?div-1:0);
  return ret;
}



int dump_fll(int f) {
  fll_t *fll = (fll_t*)0x1a100000;
  int div = 	 fll[f].cfg1.clock_out_divider;
  int mult = 	 fll[f].stat.act_mult;
  
  printf("FLL[%d]: Actual Multiplier m=%d (%x) ",f,mult,mult);
  printf("Div = %d Freq = %d MHz\n",div,
	 (REFCLK)*mult /
	 (1000*spow2(div?div-1:0)));
  printf("        locken=%d, %s gain=%d Dithering %s (0x%x.0x%x)\n",
	 fll[f].cfg1.output_lock_enable,
	 fll[f].cfg1.mode ? "normal" : "standalone", 
	 fll[f].cfg2.loop_gain,
	 fll[f].cfg2.dithering ? "on": "off",
	 fll[f].integ.state_int_part,fll[f].integ.state_fract_part
	 );

  return 0;
}
