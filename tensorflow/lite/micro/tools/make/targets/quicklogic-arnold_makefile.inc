# Settings for RISCV 32-bit MCU toolchain.
ifeq ($(TARGET), quicklogic-arnold)
  TARGET_ARCH := rv32imfcxpulpv2
  TARGET_TOOLCHAIN_ROOT := /opt/riscv/bin/
  TARGET_TOOLCHAIN_PREFIX := riscv32-unknown-elf-

  #$(eval $(call add_third_party_download,$(RISCV_TOOLCHAIN_URL),$(RISCV_TOOLCHAIN_MD5),riscv_toolchain,))
  #$(eval $(call add_third_party_download,$(SIFIVE_FE310_LIB_URL),$(SIFIVE_FE310_LIB_MD5),sifive_fe310_lib,))

  PLATFORM_FLAGS = \
    -mfdiv -D__riscv__\
	  -O3\
	  -fdata-sections\
	  -ffunction-sections\
	  -MMD\
	  -MP

  CXXFLAGS += $(PLATFORM_FLAGS) \
	-DTF_LITE_USE_GLOBAL_CMATH_FUNCTIONS\
	-DTF_LITE_USE_GLOBAL_MAX\
	-DTF_LITE_USE_GLOBAL_MIN\
    -fpermissive \
    -fno-rtti \
    --std=gnu++11

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

# These are microcontroller-specific rules for converting the ELF output
# of the linker into a binary image that can be loaded directly.
  OBJCOPY := $(TARGET_TOOLCHAIN_PREFIX)objcopy
  $(BINDIR)/%.bin: $(BINDIR)/%
		@mkdir -p $(dir $@)
		$(OBJCOPY) $< $@ -O binary
endif
