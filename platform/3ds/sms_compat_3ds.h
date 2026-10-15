/* Force-included (-include) into every sms-pc-port platform unit built for the 3DS.
 * Covers POSIX bits newlib on devkitARM lacks, without touching the PC port's sources. */
#ifndef SMS_COMPAT_3DS_H
#define SMS_COMPAT_3DS_H
#ifdef __cplusplus
#include <cerrno>
#include <ctime>
#else
#include <errno.h>
#include <time.h>
#endif

#ifndef CLOCK_MONOTONIC
#define CLOCK_MONOTONIC 1
#endif
#ifndef TIMER_ABSTIME
#define TIMER_ABSTIME 1
#endif

/* port_host.h sleeps to an absolute deadline with clock_nanosleep (newlib has none).
 * Renamed through a macro so it can never clash with a declaration in the toolchain. */
#define clock_nanosleep sms3ds_clock_nanosleep
static inline int sms3ds_clock_nanosleep(int clock, int flags, const struct timespec* req, struct timespec* rem)
{
	struct timespec now, left = *req;
	(void)rem;
	if (flags & TIMER_ABSTIME) {
		if (clock_gettime((clockid_t)clock, &now) != 0)
			return -1;
		left.tv_sec = req->tv_sec - now.tv_sec;
		left.tv_nsec = req->tv_nsec - now.tv_nsec;
		if (left.tv_nsec < 0) {
			left.tv_sec -= 1;
			left.tv_nsec += 1000000000L;
		}
		if (left.tv_sec < 0)
			return 0;
	}
	while (nanosleep(&left, &left) != 0) {
		if (errno != EINTR)
			return errno;
	}
	return 0;
}
#endif
