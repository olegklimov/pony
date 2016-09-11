#include "miniutils.h"
#include <stdarg.h>
#include <string>
#include <boost/thread/thread.hpp>
#include <boost/lexical_cast.hpp>

#ifdef WIN32
#include <Windows.h>
#endif

namespace miniutils {

std::string stdprintf(const char* fmt, ...)
{
	char buf[32768];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	buf[32768-1] = 0;
	return buf;
}

uint64_t now()
{
	boost::xtime xt;
	boost::xtime_get(&xt, boost::TIME_UTC_);
	uint64_t t = uint64_t(xt.sec)*1000000 + xt.nsec/1000;
	return t;
}

FILE* open_or_die(const std::string& filename, const char* mode)
{
	FILE* f = fopen( filename.c_str(), mode );
	if (!f)
		throw std::runtime_error("cannot open '" + filename + "' with mode '" + mode + "': " + strerror(errno) );
	return f;
}

void write_or_die(FILE* f, const void* data, uint32_t len, const std::string& errgen)
{
	assert(len);
	int r = (int) fwrite(data, len, 1, f);
	if (r!=1)
		throw std::runtime_error("cannot write into '" + errgen + "': " + strerror(errno) );
}

void read_or_die(FILE* f, void* data, uint32_t len, const std::string& errgen)
{
	assert(len);
	int r = (int) fread(data, len, 1, f);
	if (r==0)
		throw std::runtime_error("cannot read from '" + errgen + "', eof");
	if (r!=1)
		throw std::runtime_error("cannot read from '" + errgen + "': " + strerror(errno) );
}

std::string read_file(const std::string& name)
{
	FILE* f = open_or_die(name, "rb");
	off_t ret = fseek(f, 0, SEEK_END);
	if (ret==(off_t)-1)
		throw std::runtime_error("cannot stat '" + name + "': " + strerror(errno) );
	uint32_t file_size = (uint32_t) ftell(f); // ftell returns long int
	fseek(f, 0, SEEK_SET);
	std::string r;
	if (file_size==0) return r;
	r.resize(file_size);
	try {
		read_or_die(f, &r[0], file_size, name);
		fclose(f);
	} catch (...) {
		fclose(f);
		throw;
	}
	return r;
}

#ifdef WIN32
#define MS_VC_EXCEPTION 0x406d1388
typedef struct tagTHREADNAME_INFO {
	DWORD dwType; // must be 0x1000
	LPCSTR szName; // pointer to name (in same addr space)
	DWORD dwThreadID; // thread ID (-1 caller thread)
	DWORD dwFlags; // reserved for future use, most be zero
} THREADNAME_INFO;

void SetThreadName(DWORD dwThreadID, LPCTSTR szThreadName)
{
	THREADNAME_INFO info; info.dwType = 0x1000;
	info.szName = szThreadName;
	info.dwThreadID = dwThreadID; info.dwFlags = 0;
	__try { RaiseException(MS_VC_EXCEPTION, 0, sizeof(info) / sizeof(DWORD), (ULONG_PTR*)&info); }
	__except (EXCEPTION_CONTINUE_EXECUTION) {
	}
}

void set_thread_name(const char* name)
{
	SetThreadName(-1, name);
}

#else
#ifdef __linux__
#include <sys/prctl.h>
void set_thread_name(const char* name)
{
	prctl(PR_SET_NAME, name, 0, 0, 0);
}

#else
#ifdef __APPLE__
#include <pthread.h>
void set_thread_name(const char* name)
{
	int r = pthread_setname_np(name);
	if (r)
		printf("Failed to set thread name, %s\n", strerror(r));
}

#else
void set_thread_name(const char* name)
{
	#error "Your platform or compiler is not supported"
}
#endif
#endif
#endif

} // namespace
