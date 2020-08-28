# Settings for RISCV 32-bit MCU toolchain.
ifeq ($(TARGET), quicklogic-arnold)
  TARGET_ARCH := rv32imfcxpulpv2
  TARGET_TOOLCHAIN_ROOT := /opt/riscv/bin/
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-
	
	PULP_BUILDER = /home/gmartin/ETH/TRY5/arnold/pulp-builder

  PLATFORM_FLAGS = -D__riscv__z
  PLATFORM_FLAGS += -mfdiv
#  PLATFORM_FLAGS += -O3
  PLATFORM_FLAGS += -fdata-sections
  PLATFORM_FLAGS += -ffunction-sections
  PLATFORM_FLAGS += -MMD
  PLATFORM_FLAGS += -MP

  CXXFLAGS += $(PLATFORM_FLAGS)
  CXXFLAGS += -DTF_LITE_MCU_DEBUG_LOG#							 Use this when platform does not have "standard IO fucntions"
#  CXXFLAGS += -DTF_LITE_STRIP_ERROR_STRINGS
  CXXFLAGS += -DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
  CXXFLAGS += -DTF_LITE_USE_GLOBAL_MAX
  CXXFLAGS += -DTF_LITE_USE_GLOBAL_MIN
  CXXFLAGS += -fpermissive
  CXXFLAGS += -fno-rtti
  CXXFLAGS += -fno-exceptions
  CXXFLAGS += --std=gnu++11
  CXXFLAGS += -Itensorflow/lite/micro/quicklogic-arnold
  
  CCFLAGS += $(PLATFORM_FLAGS)

  BUILD_TYPE := micro

  INCLUDES += \
	-I/home/gmartin/ETH/TRY5/arnold/pulp-builder/install/include/io \
	-I/home/gmartin/ETH/TRY5/arnold/pulp-builder/install/include \
	-include /home/gmartin/ETH/TRY5/arnold/pulp-builder/pulp-rt-examples/arnold_tests/gpio_toggle/build/arnold/fc_config.h
	#-I${PULP_BUILDER}/hal/include/
	#-I${PULP_BUILDER}/archi/include/

  SRCS += tensorflow/lite/micro/quicklogic-arnold/fll.cc
  SRCS += tensorflow/lite/micro/quicklogic-arnold/gpio.cc

  MICROLITE_CC_SRCS += $(SRCS)

  MICRO_SPEECH_TEST_SRCS += 

  LDFLAGS += \
   -D__riscv__\
	-MMD\
	-MP\
	-lbench\
	-nostartfiles\
	-nostdlib\
	-Wl,--gc-sections\
	-L${PULP_BUILDER}/install/rules\
	-T../../../../../targets/quicklogic-arnold/arnold.ld\
	-L${PULP_BUILDER}/install/lib/arnold\
	-L${PULP_BUILDER}/install/lib/arnold/arnold\
	-lrt\
	-lrtio\
	-lrt\
	-lgcc
#LDFLAGS += -lstdc++


# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)/%.bin: $(BINDIR)/%
		@mkdir -p $(dir $@)
		$(OBJCOPY) $< $@ -O binary
endif
