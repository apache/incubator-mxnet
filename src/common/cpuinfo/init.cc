//Copyright (c) 2017-2018 Facebook Inc.
//Copyright (C) 2012-2017 Georgia Institute of Technology
//Copyright (C) 2010-2012 Marat Dukhan
//
//All rights reserved.
//
//Redistribution and use in source and binary forms, with or without
//    modification, are permitted provided that the following conditions are met:
//
//* Redistributions of source code must retain the above copyright notice, this
//list of conditions and the following disclaimer.
//
//* Redistributions in binary form must reproduce the above copyright notice,
//this list of conditions and the following disclaimer in the documentation
//and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
//    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
//    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
//OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#endif


#include "../cpuinfo.h"
#include "./api.h"

#ifdef __APPLE__
#include "TargetConditionals.h"
#endif


#ifdef _WIN32
static INIT_ONCE init_guard = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t init_guard = PTHREAD_ONCE_INIT;
#endif

#include <stdint.h>
#include <string.h>

#include "../cpuinfo.h"
#include "cpuid.h"
#include "api.h"
#include "utils.h"

namespace mxnet {
namespace common {
namespace cpuinfo {



bool CPUINFO_ABI cpuinfo_initialize(void) {
#if CPUINFO_ARCH_X86 || CPUINFO_ARCH_X86_64
	#if defined(__MACH__) && defined(__APPLE__)
		pthread_once(&init_guard, &cpuinfo_x86_mach_init);
	#elif defined(__linux__)
		pthread_once(&init_guard, &cpuinfo_x86_linux_init);
	#elif defined(_WIN32)
		InitOnceExecuteOnce(&init_guard, &cpuinfo_x86_windows_init, NULL, NULL);
	#else
		cpuinfo_log_error("operating system is not supported in cpuinfo");
	#endif
#else
	cpuinfo_log_error("processor architecture is not supported in cpuinfo");
#endif
	return cpuinfo_is_initialized;
}

void CPUINFO_ABI cpuinfo_deinitialize(void) {
}

void cpuinfo_x86_init_processor(struct cpuinfo_x86_processor* processor) {
	const struct cpuid_regs leaf0 = cpuid(0);
	const uint32_t max_base_index = leaf0.eax;

	const struct cpuid_regs leaf0x80000000 = cpuid(UINT32_C(0x80000000));
	const uint32_t max_extended_index =
			leaf0x80000000.eax >= UINT32_C(0x80000000) ? leaf0x80000000.eax : 0;

	if (max_base_index >= 1) {
		const struct cpuid_regs leaf1 = cpuid(1);
		processor->cpuid = leaf1.eax;

		cpuinfo_x86_detect_topology(max_base_index, max_extended_index, leaf1, &processor->topology);

	}
}

} // cpuinfo
} // common
} // mxnet