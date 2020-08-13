# Settings for RISCV 32-bit MCU toolchain.
ifeq ($(TARGET), quicklogic-arnold)
  TARGET_ARCH := rv32imfcxpulpv2
  TARGET_TOOLCHAIN_ROOT := /opt/riscv/bin/
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

  PLATFORM_FLAGS = \
    -mfdiv -D__riscv__\
	  -O3\
	  -fdata-sections\
	  -ffunction-sections\
	  -MMD\
	  -MP

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

  CCFLAGS += $(PLATFORM_FLAGS)

  BUILD_TYPE := micro

  INCLUDES += \


  MICROLITE_CC_SRCS += 

  MICRO_SPEECH_TEST_SRCS += 

  LDFLAGS += \
   -D__riscv__\
	-MMD\
	-MP\
	-lbench\
	-nostartfiles\
	-nostdlib\
	-Wl,--gc-sections\
	-L/home/gmartin/ETH/TRY5/arnold/pulp-builder/install/rules\
	-Tarnold/link.ld\
	-L/home/gmartin/ETH/TRY5/arnold/pulp-builder/install/lib/arnold\
	-L/home/gmartin/ETH/TRY5/arnold/pulp-builder/install/lib/arnold/arnold\
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
