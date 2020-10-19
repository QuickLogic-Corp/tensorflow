# Settings for RISCV 32-bit MCU toolchain.
ifeq ($(TARGET), quicklogic-arnold)
  TARGET_ARCH := rv32imfcxpulpv2
  TARGET_TOOLCHAIN_ROOT := /opt/riscv/bin/
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-
  # change PULP_BUILDER to be the location where pulp-builder is 
  PULP_BUILDER = ${PULP_SDK_HOME}

  PLATFORM_FLAGS = -D__riscv__z
  PLATFORM_FLAGS += -march=rv32imfcxpulpv2 
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
  CXXFLAGS += -I${PULP_BUILDER}/install/include
  CXXFLAGS += -I${PULP_BUILDER}/hal/include
  CXXFLAGS += -I${PULP_BUILDER}/archi/include
  CXXFLAGS += -I${PULP_BUILDER}/install/include/archi/chips/arnold
  
  CCFLAGS += $(PLATFORM_FLAGS)
  CCFLAGS += --std=gnu99
  CCFLAGS += -I${PULP_BUILDER}/install/include
  CCFLAGS += -I${PULP_BUILDER}/hal/include
  CCFLAGS += -I${PULP_BUILDER}/archi/include

  BUILD_TYPE := micro

  INCLUDES += \
	-I${PULP_BUILDER}/install/include/io \
	-I${PULP_BUILDER}/install/include


	SRCS += tensorflow/lite/micro/quicklogic-arnold/conv-simd.cc
	SRCS += tensorflow/lite/micro/quicklogic-arnold/fll.cc
	SRCS += tensorflow/lite/micro/quicklogic-arnold/gpio.cc
	SRCS += tensorflow/lite/micro/quicklogic-arnold/programFPGA.cc
	SRCS += tensorflow/lite/micro/quicklogic-arnold/rt_conf.c
	SRCS += tensorflow/lite/micro/quicklogic-arnold/rt_pad_conf.c

  MICROLITE_CC_SRCS += $(SRCS)

  MICRO_SPEECH_TEST_SRCS += 

  LDFLAGS += \
        -D__riscv__\
	-MMD\
	-MP\
	-nostartfiles\
	-nostdlib\
	-Wl,--gc-sections\
	-L${PULP_BUILDER}/install/rules\
	-T../../../../../targets/quicklogic-arnold/arnold.ld\
	-L${PULP_BUILDER}/install/lib/arnold\
	-lbench\
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
