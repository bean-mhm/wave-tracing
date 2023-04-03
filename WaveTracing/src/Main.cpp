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
    params.resX = 501;
    params.resY = 501;
    params.resZ = 501;
    params.step = 0.0025f;
    params.speed = 30.0f;
    params.damp = 1.0f;

    auto dims = params.getDimensions();

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
    initial.resize(params.resX * params.resY * params.resZ);
    for (auto& v : initial)
        v = Random::nextFloat(-0.5f, 0.5f);

    Wave3D wave(params, initial);

    float timestep = params.getMaxTimestep();
    uint32_t runs = 10;

    std::cout << "Starting the simulation...\n";

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < runs; i++)
    {
        wave.increment(timestep);
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0;

    uint64_t numPoints = params.resX * params.resY * params.resZ;
    float pointsPerSec = (float)runs * (float)numPoints / duration;

    std::cout << strFormat(
        "%u steps done in %.1f ms  -  %.1f million points / sec\n",
        runs,
        duration * 1000.0,
        pointsPerSec / 1000000.0f);

    std::cout << strFormat("sim time: %f s\n", wave.getTotalTime());

    return 0;
}
