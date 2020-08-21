/* 

    printf("Attempted to program FLL0 %d MHz\n",prog_fll(0,66,2));
*/
#include <stdio.h>

int spow2 (int exp) {
  int i, ret;
  ret = 1;
  if (exp == 0) return ret;
  else
    for (i = 0; i < exp; ret = ret * 2, i++);
  return ret;
}

int prog_fll(int fll, int fref, int mult, int div) {
  int ret = (12 * (mult+1))/spow2(div?(div-1):0);
  int i;
  i = 0x80000000 |  // 1 = Normal mode
      0x40000000 |  // 1 = FLL output gated by LOCK
      (div << 26) | 
      (0x180 << 16) | 
      mult;
  int uxconfigreg2 =
        0x00000000 |  // 1 = Dithering enabled
        0x40000000 |  // 1 = Open-loop-when-locked
        0x20000000 |  // 1 = REFCLK, 0 = DCOCLK
        (0x200 << 16) | // lock tolerance
        (0x10  << 10) | // Stable REFCLK until LOCK assert
        (0x20  << 4)  | // Unstable REFCLK until LOCK deassert
        3             ; // -lg(FLL loop gain) ie. 7 => 2^(-7) = 1/256
  int regStatusI;
  int regConfigI;
  int regConfigII;
  int regIntegrator;
  switch (fll) {
  case 0:
    *(unsigned int*)0x1a100008 = uxconfigreg2; // select ref clock as input
    *(unsigned int*)0x1a100004 = i;
    regStatusI = *(unsigned int*)0x1a100000,
	  regConfigI = *(unsigned int*)0x1a100004,
	  regConfigII = *(unsigned int*)0x1a100008,
	  regIntegrator = *(unsigned int*)0x1a10000C;
    printf("FLL0 m=%d, div= %d %08x %08x %08x %08x\n",mult,div,
	   *(unsigned int*)0x1a100000,
	   *(unsigned int*)0x1a100004,
	   *(unsigned int*)0x1a100008,
	   *(unsigned int*)0x1a10000C);
    break;
  case 1:
    *(unsigned int*)0x1a100018 = uxconfigreg2; // select ref clock as input
    *(unsigned int*)0x1a100014 = i;
    printf("FLL1=1 m=%d, div= %d %08x %08x %08x %08x\n",mult,div,
	   *(unsigned int*)0x1a100010,
	   *(unsigned int*)0x1a100014,
	   *(unsigned int*)0x1a100018,
	   *(unsigned int*)0x1a10001C);
    break;
  case 2:
    *(unsigned int*)0x1a10002C = 0x00808000;
    *(unsigned int*)0x1a100028 = 0x62004100; // select ref clock as input
    *(unsigned int*)0x1a100024 = i;
    printf("FLL2 m=%d, div= %d %08x %08x %08x %08x\n",mult,div,
	   *(unsigned int*)0x1a100020,
	   *(unsigned int*)0x1a100024,
	   *(unsigned int*)0x1a100028,
	   *(unsigned int*)0x1a10002C);
    break;
  }
  printf("FLL%d(m=%d, div=%d)\n", fll, mult, div);
  printf("StatusI=0x%08x [actual mult=%d, assuming %dMHz => %dMHz\n", regStatusI, regStatusI, fref, fref * regStatusI);
  return ret;
}
