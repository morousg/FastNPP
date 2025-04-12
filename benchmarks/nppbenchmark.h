/* Copyright 2025 Oscar Amoros Huguet
   Copyright 2025 Albert Andaluz Gonzalez
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
   

std::unordered_map<std::string, std::stringstream> benchmarkResultsText;
std::unordered_map<std::string, std::ofstream> currentFile;
// Select the path where to write the benchmark files
const std::string path{ "" };

constexpr int ITERS = 100;

bool warmup{ false };

struct BenchmarkResultsNumbers {
  float NPPelapsedTimeMax;
  float NPPelapsedTimeMin;
  float NPPelapsedTimeAcum;
  float FKelapsedTimeMax;
  float FKelapsedTimeMin;
  float FKelapsedTimeAcum;
};

template <size_t ITERATIONS> float computeVariance(const float &mean, const std::array<float, ITERATIONS> &times) {
  float sumOfDiff = 0.f;
  for (int idx = 0; idx <ITERATIONS; ++idx) {
    const float diff = times[idx] - mean;
    sumOfDiff += (diff * diff);
  }
  return sumOfDiff / (ITERATIONS - 1);
}

template <int BATCH, int ITERS, int NUM_BATCH_VALUES, const std::array<size_t, NUM_BATCH_VALUES> &batchValues>
inline void processExecution(const BenchmarkResultsNumbers &resF, const std::string &functionName,
                             const std::array<float, ITERS> &NPPelapsedTime,
                             const std::array<float, ITERS> &FKelapsedTime,
                             const std::string &variableDimension) {
  // Create 2D Table for changing types and changing batch
  const std::string fileName = functionName + std::string(".csv");
  if constexpr (BATCH == batchValues[0]) {
    if (currentFile.find(fileName) == currentFile.end()) {
      currentFile[fileName].open(path + fileName);
    }
    currentFile[fileName] << variableDimension;
    currentFile[fileName] << ", NPP MeanTime";
    currentFile[fileName] << ", NPP TimeVariance";
    currentFile[fileName] << ", NPP MaxTime";
    currentFile[fileName] << ", NPP MinTime";
    currentFile[fileName] << ", FK MeanTime";
    currentFile[fileName] << ", FK TimeVariance";
    currentFile[fileName] << ", FK MaxTime";
    currentFile[fileName] << ", FK MinTime";
    currentFile[fileName] << ", Mean Speedup";
    currentFile[fileName] << std::endl;
  }

  const bool mustStore = currentFile.find(fileName) != currentFile.end();
  if (mustStore) {
    const float NPPMean = resF.NPPelapsedTimeAcum / ITERS;
    const float FKMean = resF.FKelapsedTimeAcum / ITERS;
    const float NPPVariance = computeVariance(NPPMean, NPPelapsedTime);
    const float FKVariance = computeVariance(FKMean, FKelapsedTime);
    float meanSpeedup{0.f};
    for (int idx = 0; idx <ITERS; ++idx) {
      meanSpeedup += NPPelapsedTime[idx] / FKelapsedTime[idx];
    }
    meanSpeedup /= ITERS;

 
    currentFile[fileName] << BATCH;
    currentFile[fileName] << ", " << NPPMean;
    currentFile[fileName] << ", " << computeVariance(NPPMean, NPPelapsedTime);
    currentFile[fileName] << ", " << resF.NPPelapsedTimeMax;
    currentFile[fileName] << ", " << resF.NPPelapsedTimeMin;
    currentFile[fileName] << ", " << FKMean;
    currentFile[fileName] << ", " << computeVariance(FKMean, FKelapsedTime);
    currentFile[fileName] << ", " << resF.FKelapsedTimeMax;
    currentFile[fileName] << ", " << resF.FKelapsedTimeMin;
    currentFile[fileName] << ", " << meanSpeedup;
    currentFile[fileName] << std::endl;
  }
}
 
 
 
#define START_NPP_BENCHMARK                                                                                            \
  std::cout << "Executing " << __func__ << " fusing " << BATCH << " operations. " << (BATCH - FIRST_VALUE) / INCREMENT \
            << "/" << NUM_EXPERIMENTS << std::endl;                                                                    \
  cudaEvent_t start, stop;                                                                                             \
  BenchmarkResultsNumbers resF;                                                                                        \
  resF.NPPelapsedTimeMax = fk::minValue<float>;                                                                        \
  resF.NPPelapsedTimeMin = fk::maxValue<float>;                                                                        \
  resF.NPPelapsedTimeAcum = 0.f;                                                                                       \
  resF.FKelapsedTimeMax = fk::minValue<float>;                                                                         \
  resF.FKelapsedTimeMin = fk::maxValue<float>;                                                                         \
  resF.FKelapsedTimeAcum = 0.f;                                                                                        \
                                               \
  gpuErrchk(cudaEventCreate(&start));                                                                                  \
  gpuErrchk(cudaEventCreate(&stop));                                                                                   \
  std::array<float, ITERS> NPPelapsedTime;                                                                             \
  std::array<float, ITERS> FKelapsedTime;                                                                              \
  for (int idx = 0; idx <ITERS; ++idx) {                                                                                    \
    gpuErrchk(cudaEventRecord(start, compute_stream));
 

 
#define STOP_NPP_START_FK_BENCHMARK                                                                                    \
  gpuErrchk(cudaEventRecord(stop, compute_stream));                                                                            \
  gpuErrchk(cudaEventSynchronize(stop));                                                                               \
  gpuErrchk(cudaEventElapsedTime(&NPPelapsedTime[idx], start, stop));                                                    \
  resF.NPPelapsedTimeMax = resF.NPPelapsedTimeMax < NPPelapsedTime[idx] ? NPPelapsedTime[idx] : resF.NPPelapsedTimeMax;    \
  resF.NPPelapsedTimeMin = resF.NPPelapsedTimeMin > NPPelapsedTime[idx] ? NPPelapsedTime[idx] : resF.NPPelapsedTimeMin;    \
  resF.NPPelapsedTimeAcum += NPPelapsedTime[idx];                                                                        \
  gpuErrchk(cudaEventRecord(start, compute_stream));
 

 
#define STOP_FK_BENCHMARK                                                                                              \
  gpuErrchk(cudaEventRecord(stop, compute_stream));                                                                            \
  gpuErrchk(cudaEventSynchronize(stop));                                                                               \
  gpuErrchk(cudaEventElapsedTime(&FKelapsedTime[idx], start, stop));                                                     \
  resF.FKelapsedTimeMax = resF.FKelapsedTimeMax < FKelapsedTime[idx] ? FKelapsedTime[idx] : resF.FKelapsedTimeMax;         \
  resF.FKelapsedTimeMin = resF.FKelapsedTimeMin > FKelapsedTime[idx] ? FKelapsedTime[idx] : resF.FKelapsedTimeMin;         \
  resF.FKelapsedTimeAcum += FKelapsedTime[idx];                                                                          \
  }                                                                                                                  \
processExecution<BATCH, ITERS, batchValues.size(), batchValues>(                               \
      resF, __func__, NPPelapsedTime, FKelapsedTime ,VARIABLE_DIMENSION);
 
 
#define CLOSE_BENCHMARK                                                                                                \
  for (auto &&[_, file] : currentFile) {                                                                               \
    file.close();                                                                                                      \
  }
 
template <int ITERS>
struct BenchmarkTemp {
    BenchmarkResultsNumbers resF;
    std::chrono::steady_clock::time_point startTime;
    std::array<float, ITERS> NPPelapsedTime;
    std::array<float, ITERS> FastNPPelapsedTime;
};

#ifdef ENABLE_BENCHMARK
template <int ITERS, int BATCH, int FIRST_VALUE, int INCREMENT, int NUM_EXPERIMENTS>
BenchmarkTemp<ITERS> initCPUBenchmark(const std::string & functionName, const std::string & variableDimension) {
    std::cout << "Executing " << functionName << " fusing " << BATCH << " operations. "
        << ((BATCH - FIRST_VALUE) / INCREMENT) + 1 << "/" << NUM_EXPERIMENTS << std::endl;
    BenchmarkTemp<ITERS> benchmarkTemp;
    benchmarkTemp.resF.NPPelapsedTimeMax = fk::minValue<float>;
    benchmarkTemp.resF.NPPelapsedTimeMin = fk::maxValue<float>;
    benchmarkTemp.resF.NPPelapsedTimeAcum = 0.f;
    benchmarkTemp.resF.FKelapsedTimeMax = fk::minValue<float>;
    benchmarkTemp.resF.FKelapsedTimeMin = fk::maxValue<float>;
    benchmarkTemp.resF.FKelapsedTimeAcum = 0.f;

    return benchmarkTemp;
}

template <int ITERS>
void startNPP_CPU(BenchmarkTemp<ITERS>&benchmarkTemp) {
    benchmarkTemp.startTime = std::chrono::high_resolution_clock::now();
}

template <int ITERS>
void stopNPP_startFastNPP_CPU(BenchmarkTemp<ITERS>&benchmarkTemp, const int& i) {
    const auto endTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> elapsedTime = endTime - benchmarkTemp.startTime;
    benchmarkTemp.NPPelapsedTime[i] = elapsedTime.count();
    benchmarkTemp.resF.NPPelapsedTimeMax = benchmarkTemp.resF.NPPelapsedTimeMax < benchmarkTemp.NPPelapsedTime[i] ? benchmarkTemp.NPPelapsedTime[i] : benchmarkTemp.resF.NPPelapsedTimeMax;
    benchmarkTemp.resF.NPPelapsedTimeMin = benchmarkTemp.resF.NPPelapsedTimeMin > benchmarkTemp.NPPelapsedTime[i] ? benchmarkTemp.NPPelapsedTime[i] : benchmarkTemp.resF.NPPelapsedTimeMin;
    benchmarkTemp.resF.NPPelapsedTimeAcum += benchmarkTemp.NPPelapsedTime[i];

    benchmarkTemp.startTime = std::chrono::high_resolution_clock::now();
}

template <int ITERS>
void stopFastNPP_CPU(BenchmarkTemp<ITERS>&benchmarkTemp, const int& i) {
    const auto endTime = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<float, std::milli> elapsedTime = endTime - benchmarkTemp.startTime;
    benchmarkTemp.FastNPPelapsedTime[i] = elapsedTime.count();
    benchmarkTemp.resF.FKelapsedTimeMax = benchmarkTemp.resF.FKelapsedTimeMax < benchmarkTemp.FastNPPelapsedTime[i] ? benchmarkTemp.FastNPPelapsedTime[i] : benchmarkTemp.resF.FKelapsedTimeMax;
    benchmarkTemp.resF.FKelapsedTimeMin = benchmarkTemp.resF.FKelapsedTimeMin > benchmarkTemp.FastNPPelapsedTime[i] ? benchmarkTemp.FastNPPelapsedTime[i] : benchmarkTemp.resF.FKelapsedTimeMin;
    benchmarkTemp.resF.FKelapsedTimeAcum += benchmarkTemp.FastNPPelapsedTime[i];
}

void stopCPUBenchmark() {
    for (auto&& [_, file] : currentFile) {
        file.close();
    }
}

#else

template <int ITERS>
void startNPP_CPU(BenchmarkTemp<ITERS>&benchmarkTemp) {}

template <int ITERS>
void stopNPP_startFastNPP_CPU(BenchmarkTemp<ITERS>&benchmarkTemp, const int& i) {}

template <int ITERS>
void stopFastNPP_CPU(BenchmarkTemp<ITERS>&benchmarkTemp, const int& i) {}

void stopCPUBenchmark() {}

#endif // ENABLE_BENCHMARK