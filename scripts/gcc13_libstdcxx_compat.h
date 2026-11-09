#ifndef REAL2SIM_GCC13_LIBSTDCXX_COMPAT_H
#define REAL2SIM_GCC13_LIBSTDCXX_COMPAT_H

// GCC 13's <iostream> otherwise references a GLIBCXX_3.4.32 initializer.
// Load the generated configuration first, then select the traditional
// initializer so the CUDA extensions remain usable with the host's older
// libstdc++ after the GCC 13 build toolchain leaves PATH.
#include <bits/c++config.h>
#undef _GLIBCXX_USE_INIT_PRIORITY_ATTRIBUTE
#define _GLIBCXX_USE_INIT_PRIORITY_ATTRIBUTE 0

#endif
