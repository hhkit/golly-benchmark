#pragma once
#include "defines.h"

#if defined(_WIN32) || defined(_MSC_VER)
  #include <windows.h>
#elif defined(__APPLE__) && defined(__MACH__)
  // http://developer.apple.com/qa/qa2004/qa1398.html
  #include <mach/mach_time.h>
#else // Linux
  #ifndef AF_DOC
    #include <sys/time.h>
  #endif
#endif

namespace af {

/// Internal timer object
    typedef struct timer {
    #if defined(_WIN32) || defined(_MSC_VER)
      LARGE_INTEGER val;
    #elif defined(__APPLE__) && defined(__MACH__)
      uint64_t val;
    #else // Linux
      struct timeval val;
    #endif

    AFAPI static timer start();

    AFAPI static double stop();
    AFAPI static double stop(timer start);

} timer;

AFAPI double timeit(void(*fn)());
}
