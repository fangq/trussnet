// SPDX-License-Identifier: GPL-3.0-or-later
//
// trussnet -- Copyright (C) 2026  Qianqian Fang <q.fang at neu.edu>
//
// tn_log.cpp -- see tn_log.h.

#include "tn_log.h"

#include <cstdarg>

namespace tn {
namespace {
LogWriter g_writer = nullptr;
}  // namespace

void set_log_writer(LogWriter writer) {
    g_writer = writer;
}

int log_fprintf(std::FILE* stream, const char* fmt, ...) {
    std::va_list ap;
    va_start(ap, fmt);

    // Only console streams go to the host writer; real files (debug dumps) keep
    // writing to their FILE*.
    if (g_writer && (stream == stderr || stream == stdout)) {
        // Format into a buffer, then hand the text to the host (e.g. mexPrintf).
        char buf[8192];
        int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        g_writer(buf);
        return n;
    }

    int n = std::vfprintf(stream, fmt, ap);
    va_end(ap);
    return n;
}

}  // namespace tn
