/* sys/uio.h for the 3DS: the iovec type only (newlib has no readv/writev).
 * Adapted from chasem-dev/sms-switch (MIT). Uses the toolchain's header when present. */
#ifndef SMS_3DS_SYS_UIO_H
#define SMS_3DS_SYS_UIO_H
#if !defined(SMS_3DS_FORCE_SHIMS) && defined(__has_include_next)
#if __has_include_next(<sys/uio.h>)
#include_next <sys/uio.h>
#define SMS_3DS_HAVE_REAL_UIO 1
#endif
#endif
#ifndef SMS_3DS_HAVE_REAL_UIO
#include <stddef.h>
struct iovec {
	void* iov_base;
	size_t iov_len;
};
#endif
#endif
