// Centralized message output for trussnet.
//
// All status/progress text in the library is emitted with TN_FPRINTF (a
// drop-in for std::fprintf) instead of std::fprintf directly, so the embedding
// host can redirect it. The CLI keeps the default (writes to the given stdio
// stream); the MATLAB/Octave MEX installs a writer that calls mexPrintf(), so
// messages land in the MATLAB console rather than the OS terminal that launched
// MATLAB. Because b2m_core is one prebuilt static library shared by the CLI,
// the MEX and the Python module, the redirection is a *runtime* writer hook
// rather than a compile-time macro (which could not reach the prebuilt objects).
#ifndef TN_LOG_H
#define TN_LOG_H

#include <cstdio>

namespace tn {

// Writer hook: receives already-formatted text. nullptr (default) -> stdio.
using LogWriter = void (*)(const char* text);
void set_log_writer(LogWriter writer);

// printf-style; routes to the installed writer, else std::fprintf(stream, ...).
#if defined(__GNUC__)
    __attribute__((format(printf, 2, 3)))
#endif
int log_fprintf(std::FILE* stream, const char* fmt, ...);

}  // namespace tn

// Drop-in replacement for std::fprintf(stream, fmt, ...).
#define TN_FPRINTF(stream, ...) ::tn::log_fprintf((stream), __VA_ARGS__)

#endif  // TN_LOG_H
