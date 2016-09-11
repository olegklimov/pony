#ifndef MINIUTILS_H
#define MINIUTILS_H
//#include <stdint.h>
#include <cstdint>
#include <string>
#include <stdexcept>
#include <map>

namespace miniutils {

uint64_t now();
std::string stdprintf(const char* fmt, ...);

std::string read_file(const std::string& name);
void read_or_die(FILE* f, void* data, uint32_t len, const std::string& errgen);
void write_or_die(FILE* f, const void* data, uint32_t len, const std::string& errgen);
FILE* open_or_die(const std::string& filename, const char* mode);

void set_thread_name(const char*); // for debugging

} // namespace

#endif // MINIUTILS_H
