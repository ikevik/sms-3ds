// POSIX pieces the sms-pc-port platform layer uses that newlib on devkitARM lacks.
// Every definition is weak, so a symbol the toolchain does provide wins at link time.
// Structure follows chasem-dev/sms-switch's switch/src/posix_shims.cpp (MIT).
#include <3ds.h>
#include <errno.h>
#include <malloc.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <sys/mman.h>
#include <sys/utsname.h>

#define WEAK __attribute__((weak))

// --- dlfcn: no dynamic loading; the port's optional SDL lookups simply fail -----
static const char* g_dlerror;
extern "C" WEAK void* dlopen(const char*, int)
{
	g_dlerror = "dynamic loading is not available on the 3DS";
	return NULL;
}
extern "C" WEAK int dlclose(void*) { return 0; }
extern "C" WEAK void* dlsym(void*, const char*)
{
	g_dlerror = "symbol not available on the 3DS";
	return NULL;
}
extern "C" WEAK char* dlerror(void)
{
	const char* e = g_dlerror;
	g_dlerror = NULL;
	return (char*)e;
}
extern "C" WEAK int dladdr(const void*, Dl_info*) { return 0; }

// --- execinfo: no unwinder is wired up yet -----------------------------------------
extern "C" WEAK int backtrace(void**, int) { return 0; }
extern "C" WEAK char** backtrace_symbols(void* const*, int) { return NULL; }
extern "C" WEAK void backtrace_symbols_fd(void* const*, int, int) {}

// --- mmap: anonymous, heap-backed ----------------------------------------------------
// A request for a fixed address (MEM1 at 0x80000000, the GX register window at
// 0xCC000000) cannot be honoured on Horizon and fails, which sends the PC port down its
// own fallback: a plain heap block.
extern "C" WEAK void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t)
{
	(void)prot;
	if ((flags & (MAP_FIXED | MAP_FIXED_NOREPLACE)) && addr)
		return MAP_FAILED;
	if (fd != -1 || !(flags & MAP_ANONYMOUS) || length == 0) {
		errno = EINVAL;
		return MAP_FAILED;
	}
	size_t rounded = (length + 0xFFF) & ~(size_t)0xFFF;
	void* p = memalign(0x1000, rounded);
	if (!p) {
		errno = ENOMEM;
		return MAP_FAILED;
	}
	memset(p, 0, rounded); // anonymous mappings are zero-filled
	return p;
}
extern "C" WEAK int munmap(void* addr, size_t) { free(addr); return 0; }
extern "C" WEAK int mprotect(void*, size_t, int) { return 0; }
extern "C" WEAK int madvise(void*, size_t, int) { return 0; }

// --- identity ------------------------------------------------------------------------
extern "C" WEAK int uname(struct utsname* name)
{
	memset(name, 0, sizeof *name);
	strcpy(name->sysname, "Horizon");
	strcpy(name->nodename, "3ds");
	strcpy(name->release, "libctru");
	strcpy(name->version, "devkitARM");
	strcpy(name->machine, "armv6k");
	return 0;
}

extern "C" WEAK long sysconf(int name)
{
	// _SC_PAGESIZE is 8 on newlib/glibc-style numbering, but only the page size and core
	// count matter to the port; answer those and refuse the rest.
	enum { kPageSize = 8, kProcsConf = 83, kProcsOnln = 84 };
	switch (name) {
	case kPageSize: return 0x1000;
	case kProcsConf: return 4;
	case kProcsOnln: return 2; // application cores: 2 on New 3DS (core 0 + core 2), 1 on Old
	default: errno = EINVAL; return -1;
	}
}

// --- files ----------------------------------------------------------------------------
// newlib has no pread: seek, read and restore under one lock (the disc is read from
// several threads). A failed read is tried again: the disc reader (sms-pc-port's gcdisc.cpp)
// gives up on the first failure and returns nothing, and the game then decompresses what the
// buffer held before (crash dump 42: Ricco Harbor's second episode, four of its models and
// animations unreadable after a long session on a console). Each failure is logged.
static LightLock g_pread_lock = 1;  // 1: unlocked
extern "C" WEAK ssize_t pread(int fd, void* buf, size_t count, off_t offset)
{
	LightLock_Lock(&g_pread_lock);
	ssize_t n = -1;
	int err = 0, tries = 0;
	off_t was = lseek(fd, 0, SEEK_CUR);
	for (; tries < 8; tries++) {
		if (tries) svcSleepThread(2000000LL * tries);  // 2, 4, ... ms
		n = -1;
		if (lseek(fd, offset, SEEK_SET) != (off_t)-1) n = read(fd, buf, count);
		if (n > 0 || count == 0) break;
		err = errno;
		if (n == 0) {  // nothing at or past the end: as asked, unless the file is longer
			struct stat st;
			if (fstat(fd, &st) == 0 && offset >= st.st_size) break;
		}
	}
	if (was != (off_t)-1) lseek(fd, was, SEEK_SET);
	LightLock_Unlock(&g_pread_lock);
	if (tries)
		fprintf(stderr, "[file] read of %lu bytes at %lld %s after %d tries (errno %d)\n", (unsigned long)count,
		        (long long)offset, n > 0 ? "worked" : "FAILED", tries + (n > 0 ? 1 : 0), err);
	return n;
}

// --- PC-port hooks with no 3DS equivalent --------------------------------------------
extern "C" WEAK void port_window_icon_init(void) {}
