// Hardware-specific support functions that MUST be customized:
#include "apb_conv2d.h"
#define I2CSPEED 4000
void I2C_delay(void);
static inline int read_SCL(void)  { return efpga->i2c & 2 ? 1 : 0;}
static inline int read_SDA(void)  { return efpga->i2c & 1 ? 1 : 0;}
static inline void set_SCL(void)   { efpga->i2c = efpga->i2c | 2;} 
static inline void clear_SCL(void) { efpga->i2c = efpga->i2c & 1;} 
static inline void clear_SDA(void) { efpga->i2c = efpga->i2c & 2;}
static inline void set_SDA(void)   { efpga->i2c = efpga->i2c | 1;}

int started = 0; // global data

void i2c_start_cond(void) {
  if (started) { 
// if started, do a restart condition
    // set SDA to 1
    set_SDA();
    I2C_delay();
    set_SCL();
    while (read_SCL() == 0) { // Clock stretching
      // You should add timeout to this loop
    }
    
    // Repeated start setup time, minimum 4.7us
    I2C_delay();
  }
  
  if (read_SDA() == 0) {
    started = 0;
  }
  else
    started = 1;
  
  // SCL is high, set SDA from 1 to 0.
  clear_SDA();
  I2C_delay();
  clear_SCL();

}

void i2c_stop_cond(void) {
  // set SDA to 0
  clear_SDA();
  I2C_delay();
  
  set_SCL();
  // Clock stretching
  while (read_SCL() == 0) {
    // add timeout to this loop.
  }
  
  // Stop bit setup time, minimum 4us
  I2C_delay();
  
  // SCL is high, set SDA from 0 to 1
  set_SDA();
  I2C_delay();
  
  if (read_SDA() == 0) {
    started = 0;
  }
  
  started = 0;
}

// Write a bit to I2C bus
void i2c_write_bit(int bit) {
  if (bit) {
    set_SDA();
  } else {
    clear_SDA();
  }
  
  // SDA change propagation delay
  I2C_delay();
  
  // Set SCL high to indicate a new valid SDA value is available
  set_SCL();
  
  // Wait for SDA value to be read by slave, minimum of 4us for standard mode
  I2C_delay();
  
  while (read_SCL() == 0) { // Clock stretching
    // You should add timeout to this loop
  }
  
  // SCL is high, now data is valid
  // If SDA is high, check that nobody else is driving SDA
  if (bit && (read_SDA() == 0)) {
    started = 0;
  }
  
  // Clear the SCL to low in preparation for next change
  clear_SCL();
}

// Read a bit from I2C bus
int i2c_read_bit(void) {
  int bit;
  
  // Let the slave drive data
  set_SDA();
  
  // Wait for SDA value to be written by slave, minimum of 4us for standard mod
  I2C_delay();
  // Set SCL high to indicate a new valid SDA value is available
  set_SCL();
  
  while (read_SCL() == 0) { // Clock stretching
    // You should add timeout to this loop
  }
  
  // Wait for SDA value to be written by slave, minimum of 4us for standard mode
  I2C_delay();
  
  // SCL is high, read out bit
  bit = read_SDA();
  
  // Set SCL low in preparation for next operation
  clear_SCL();
  
  return bit;
}

// Write a byte to I2C bus. Return 0 if ack by the slave.
int i2c_write_byte(int send_start,
		    int send_stop,
		    unsigned char byte) {
  unsigned bit;
  int     nack;
  
  if (send_start) {
    i2c_start_cond();
  }
  
  for (bit = 0; bit < 8; ++bit) {
    i2c_write_bit((byte & 0x80) != 0);
    byte <<= 1;
  }
  
  nack = i2c_read_bit();
  
  if (send_stop) {
    i2c_stop_cond();
  }
  
  return nack;
}

// Read a byte from I2C bus
unsigned char i2c_read_byte(int nack, int send_stop) {
  unsigned char byte = 0;
  unsigned char bit;
  
  for (bit = 0; bit < 8; ++bit) {
    byte = (byte << 1) | i2c_read_bit();
  }
  
  i2c_write_bit(nack);
  
  if (send_stop) {
    i2c_stop_cond();
  }
  
  return byte;
}

void I2C_delay(void) { 
  volatile int v;
  int i;
  
  for (i = 0; i < I2CSPEED / 2; ++i) {
    v;
  }
}
void i2c_delay(int v) {
    for (int i = 0; i < v; i++) {
      I2C_delay();
    }
}

void i2c_16write8(char dev, int addr, char data) {
  int count = 0;
  int nak = 1;
  while ((count < 10) && nak) {
    nak = i2c_write_byte (1,0,dev << 1);
    count++;
  }
    i2c_write_byte (0,0,(addr >> 8) & 0xff);   //HM01B0_BLC_TCT_REG = 0x20
    i2c_write_byte (0,0,(addr & 0xff));
    i2c_write_byte (0,1,data);
}
char i2c_16read8 (char dev, int addr) {
    i2c_write_byte (1,0,dev << 1); // Write BLC_TCT_REG
    i2c_write_byte (0,0,(addr >> 8) & 0xff);   //HM01B0_BLC_TCT_REG = 0x20
    i2c_write_byte (0,1,(addr & 0xff));
    i2c_write_byte (1,0,((dev << 1) | 1));  // new start bit for read
    return i2c_read_byte (1,1);
}
int i2c_16read16 (char dev, int addr) {
  
  int ret;
  i2c_write_byte (1,0,dev << 1); 
  i2c_write_byte (0,0,(addr >> 8) & 0xff);   //HM01B0_BLC_TCT_REG = 0x20
  i2c_write_byte (0,1,(addr & 0xff));
  i2c_write_byte (1,0,((dev << 1) | 1));  // new start bit for read
  ret = (i2c_read_byte(0,0)) << 8;
  ret |= i2c_read_byte (1,1);
  return ret;
}
