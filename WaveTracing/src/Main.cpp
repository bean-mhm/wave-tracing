#include <iostream>
#include <vector>
#include <chrono>

#include "Wave3D.h"

#include "Utils/Random.h"
#include "Utils/Misc.h"

int main()
{
    WaveParams params;
    params.resX = 201;
    params.resY = 201;
    params.resZ = 201;
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

    std::vector<float> initial;
    initial.resize(params.resX * params.resY * params.resZ);
    for (auto& v : initial)
        v = Random::nextFloat(-0.1f, 0.1f);

    Wave3D wave(params, initial);
    float timestep = params.getMaxTimestep();
    uint32_t runs = 500;

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < runs; i++)
    {
        wave.increment(timestep);
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0f;

    std::cout << strFormat(
        "%u steps done in %.1f ms (%.1f steps/sec)\n",
        runs,
        ms,
        (float)runs / (ms / 1000.0f));

    std::cout << strFormat("sim time: %f s\n", wave.getTotalTime());

    return 0;
}
