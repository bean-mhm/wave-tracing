#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <thread>

#include "Wave3D.h"

#include "Utils/Random.h"
#include "Utils/Misc.h"

int main()
{
    WaveParams params;
    params.resX = 701;
    params.resY = 701;
    params.resZ = 701;
    params.step = 0.0025f;
    params.speed = 30.0f;
    params.damp = 1.0f;

    uint64_t numPoints = params.resX * params.resY * params.resZ;
    auto dims = params.getDimensions();

    float timestep = params.getMaxTimestep();
    uint32_t runs = 20;

    std::cout << strFormat(
        "resolution:      %u x %u x %u\n"
        "step size:       %.7f m\n"
        "dimensions:      %.7f m x %.7f m x %.7f m\n"
        "volume:          %.7f m^3\n"
        "speed:           %.7f m/s\n"
        "max timestep:    %.7f s\n"
        "min wavelength:  %.7f m\n"
        "max frequency:   %.2f hz\n"
        "damp:            %.2f\n\n",
        params.resX, params.resY, params.resZ,
        params.step,
        dims[0], dims[1], dims[2],
        params.getVolume(),
        params.speed,
        params.getMaxTimestep(),
        params.getMinWavelength(),
        params.getMaxFrequency(),
        params.damp);

    std::cout << "Making randomized buffer...\n";

    std::vector<float> initial;
    initial.resize(numPoints);

#pragma omp parallel for
    for (int i = 0; i < numPoints; i++)
    {
        initial[i] = Random::nextFloat(-0.5f, 0.5f);
    }

    Wave3D wave(params, initial);

    std::cout << "Starting the simulation...\n";
    auto start = std::chrono::high_resolution_clock::now();

    for (uint32_t i = 0; i < runs; i++)
    {
        wave.increment(timestep);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;
    float pointsPerSec = (float)runs * (float)numPoints / duration;

    std::cout << strFormat(
        "%u steps done in %.1f ms  -  %.1f million points / sec\n",
        runs,
        duration * 1000.0,
        pointsPerSec / 1000000.0f);

    std::cout << strFormat("sim time: %f s\n", wave.getTotalTime());

    return 0;
}
