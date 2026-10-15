/* sys/mman.h for the 3DS: anonymous mappings only, served from the heap by
 * platform/3ds/posix_shims.cpp. The PC port asks mmap for MEM1 at 0x80000000 first,
 * which fails here, then falls back to an ordinary block; that fallback is the path
 * the 3DS uses. Adapted from chasem-dev/sms-switch (MIT). */
#ifndef SMS_3DS_SYS_MMAN_H
#define SMS_3DS_SYS_MMAN_H
#if !defined(SMS_3DS_FORCE_SHIMS) && defined(__has_include_next)
#if __has_include_next(<sys/mman.h>)
#include_next <sys/mman.h>
#define SMS_3DS_HAVE_REAL_MMAN 1
#endif
#endif
#ifndef SMS_3DS_HAVE_REAL_MMAN
#include <stddef.h>
#include <sys/types.h>
#ifdef __cplusplus
extern "C" {
#endif
#define PROT_NONE 0
#define PROT_READ 1
#define PROT_WRITE 2
#define PROT_EXEC 4
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS
#define MAP_NORESERVE 0x4000
#define MAP_FIXED_NOREPLACE 0x100000
#define MAP_FAILED ((void*)-1)
#define MADV_NORMAL 0
#define MADV_DONTNEED 4
void* mmap(void* addr, size_t length, int prot, int flags, int fd, off_t offset);
int munmap(void* addr, size_t length);
int mprotect(void* addr, size_t length, int prot);
int madvise(void* addr, size_t length, int advice);
#ifdef __cplusplus
}
#endif
#endif
#endif
