#---------------------------------------------------------------------------------
# Super Mario Sunshine 3DS port - devkitARM / libctru / citro3d build
#---------------------------------------------------------------------------------
.SUFFIXES:

ifeq ($(strip $(DEVKITARM)),)
$(error "Please set DEVKITARM in your environment. export DEVKITARM=<path to>devkitARM")
endif

TOPDIR ?= $(CURDIR)
include $(DEVKITARM)/3ds_rules

TARGET      := sms3ds
BUILD       := build3ds
# MODE=stage0 (default): disc/pad bring-up screen. MODE=game: link the real game
# (needs scripts/build_game_lib.sh, build_platform_lib.sh, build_gx_lib.sh, link_test.sh first).
MODE        ?= stage0
ifeq ($(MODE),game)
SOURCES     := source_game platform/3ds/pica
else
SOURCES     := source platform/3ds
endif
DATA        := data
INCLUDES    := include platform/3ds
GRAPHICS    := gfx
ROMFS       := romfs
GFXBUILD    := $(ROMFS)/gfx

APP_TITLE       := Super Mario Sunshine
APP_DESCRIPTION := Native 3DS port (WIP) - requires your own GMSE01 disc image
APP_AUTHOR      := ikevik
# ICON := gfx/icon.png   (uncomment once you have a 48x48 icon)

# New 3DS strongly recommended: see docs/PORTING.md
ARCH     := -march=armv6k -mtune=mpcore -mfloat-abi=hard -mtp=soft
CFLAGS   := -g -Wall -O2 -mword-relocations -ffunction-sections $(ARCH)
CFLAGS   += $(INCLUDE) -D__3DS__ -DSMS_PLATFORM_3DS=1
CXXFLAGS := $(CFLAGS) -fno-rtti -fno-exceptions -std=gnu++17
ASFLAGS  := -g $(ARCH)
LDFLAGS   = -specs=3dsx.specs -g $(ARCH) -Wl,-Map,$(notdir $*.map)
ifeq ($(MODE),game)
# Route the game's renamed operator new/new[] through platform/3ds/game_new_3ds.cpp,
# which logs allocations the game heap could not satisfy.
SMSLIBS  := -Wl,--wrap=__sms_game_new -Wl,--wrap=__sms_game_new_array
# The game, GX and pc-port platform units use 4-byte enums (GameCube ABI); libctru, newlib
# and libstdc++ use variable-size ones. Expected, and checked where structs cross over.
SMSLIBS  += -Wl,--no-enum-size-warning
SMSLIBS  += -Wl,--whole-archive $(TOPDIR)/build/libsms_game.a $(TOPDIR)/build/libsms_platform.a $(TOPDIR)/build/libsms_gx.a -Wl,--no-whole-archive $(TOPDIR)/build/linktest/gx_stubs.o
endif
LIBS     := $(SMSLIBS) -lcitro3d -lctru -lm
LIBDIRS  := $(PORTLIBS) $(CTRULIB)

#---------------------------------------------------------------------------------
ifneq ($(BUILD),$(notdir $(CURDIR)))
export OUTPUT := $(CURDIR)/$(TARGET)
export TOPDIR := $(CURDIR)
export VPATH  := $(foreach dir,$(SOURCES),$(CURDIR)/$(dir)) \
                 $(foreach dir,$(GRAPHICS),$(CURDIR)/$(dir)) \
                 $(foreach dir,$(DATA),$(CURDIR)/$(dir))
export DEPSDIR := $(CURDIR)/$(BUILD)

CFILES    := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.c)))
CPPFILES  := $(foreach dir,$(SOURCES),$(notdir $(wildcard $(dir)/*.cpp)))
PICAFILES := $(foreach dir,shaders,$(notdir $(wildcard $(dir)/*.v.pica)))
export VPATH += $(CURDIR)/shaders

# platform/3ds/*.cpp belong to the game build (libsms_platform.a), not the bring-up screen.
ifneq ($(MODE),game)
CPPFILES :=
endif

export LD := $(if $(CPPFILES),$(CXX),$(CC))
export OFILES_SOURCES := $(CPPFILES:.cpp=.o) $(CFILES:.c=.o)
export OFILES_BIN := $(addsuffix .o,$(PICAFILES:.v.pica=.shbin))
export OFILES := $(OFILES_BIN) $(OFILES_SOURCES)
export HFILES := $(PICAFILES:.v.pica=_shbin.h)
export INCLUDE := $(foreach dir,$(INCLUDES),-I$(CURDIR)/$(dir)) \
                  $(foreach dir,$(LIBDIRS),-I$(dir)/include) \
                  -I$(CURDIR)/$(BUILD)
export LIBPATHS := $(foreach dir,$(LIBDIRS),-L$(dir)/lib)

ifeq ($(strip $(ICON)),)
	icons := $(wildcard *.png)
	ifneq (,$(findstring $(TARGET).png,$(icons)))
		export APP_ICON := $(TOPDIR)/$(TARGET).png
	else
		ifneq (,$(findstring icon.png,$(icons)))
			export APP_ICON := $(TOPDIR)/icon.png
		endif
	endif
else
	export APP_ICON := $(TOPDIR)/$(ICON)
endif

ifeq ($(strip $(NO_SMDH)),)
	export _3DSXFLAGS += --smdh=$(CURDIR)/$(TARGET).smdh
endif

.PHONY: all clean
all: $(BUILD)
	@$(MAKE) --no-print-directory -C $(BUILD) -f $(CURDIR)/Makefile
clean:
	@echo clean ...
	@rm -fr $(BUILD) $(TARGET).3dsx $(OUTPUT).smdh $(TARGET).elf
$(BUILD):
	@mkdir -p $@
else
DEPENDS := $(OFILES:.o=.d)
$(OUTPUT).3dsx : $(OUTPUT).elf $(OUTPUT).smdh
$(OUTPUT).elf  : $(OFILES)

# Sources include generated shader headers (e.g. gx_pass_shbin.h): build those first, or a
# parallel make can compile a source before its header exists.
$(OFILES_SOURCES) : $(HFILES)

# main.cpp shows the build's date and time (__DATE__/__TIME__): rebuild it every time, so the stamp
# names this build and not the last one that happened to change main.cpp.
ifneq ($(filter main.o,$(OFILES_SOURCES)),)
main.o : FORCE
FORCE:
.PHONY: FORCE
endif

# Compile .v.pica vertex shaders to embeddable .shbin objects
%.shbin.o %_shbin.h : %.shbin
	@echo $(notdir $<)
	@$(bin2o)

-include $(DEPENDS)
endif
