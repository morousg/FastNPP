/* Copyright 2024 Albert Andaluz Gonzalez

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

 
#include <fused_kernel/core/utils/vlimits.h>

#include <array>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <iostream>
#include <chrono>

#include <npp.h>

enum CHANNEL { RED, GREEN, BLUE };

constexpr inline void nppAssert(NppStatus code, const char* file, int line, bool abort = true) {
    if (code != NPP_SUCCESS) {
        std::cout << "NPP failure: "
            << " File: " << file << " Line:" << line << std::endl;
        if (abort)
            throw std::exception();
    }
}

#define NPP_CHECK(ans) { nppAssert((ans), __FILE__, __LINE__, true); }

NppStreamContext initNppStreamContext(const cudaStream_t& stream) {
    int device = 0;
    int ccmajor = 0;
    int ccminor = 0;
    uint flags;

    cudaDeviceProp prop;
    gpuErrchk(cudaGetDevice(&device)) gpuErrchk(cudaGetDeviceProperties(&prop, device));
    gpuErrchk(cudaDeviceGetAttribute(&ccmajor, cudaDevAttrComputeCapabilityMinor, device));
    gpuErrchk(cudaDeviceGetAttribute(&ccminor, cudaDevAttrComputeCapabilityMajor, device));
    gpuErrchk(cudaStreamGetFlags(stream, &flags));
    NppStreamContext nppstream = { stream,
                                  device,
                                  prop.multiProcessorCount,
                                  prop.maxThreadsPerMultiProcessor,
                                  prop.maxThreadsPerBlock,
                                  prop.sharedMemPerBlock,
                                  ccmajor,
                                  ccminor,
                                  flags };
    return nppstream;
}

template <size_t START_VALUE, size_t INCREMENT, std::size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> generate_sequence(std::index_sequence<Is...>) {
  return std::array<size_t, sizeof...(Is)>{(START_VALUE + (INCREMENT * Is))...};
}

template <size_t START_VALUE, size_t INCREMENT, size_t NUM_ELEMS>
constexpr std::array<size_t, NUM_ELEMS> arrayIndexSecuence =
    generate_sequence<START_VALUE, INCREMENT>(std::make_index_sequence<NUM_ELEMS>{});



