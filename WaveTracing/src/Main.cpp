#include <iostream>
#include <vector>
#include <chrono>

#include "Wave3D.h"

#include "Utils/Random.h"
#include "Utils/Misc.h"

int main()
{
    WaveParams params;
    params.res = {
        601,
        601,
        601
    };
    params.step = 0.0025f;
    params.speed = 30.0f;
    params.damp = 1.0f;

    auto dims = params.getDimensions();

    std::cout << strFormat(
        "resolution:      %u x %u x %u\n"
        "sub resolution:  %u x %u x %u\n"
        "use sub-grid:    %s\n"
        "step size:       %.7f m\n"
        "dimensions:      %.7f m x %.7f m x %.7f m\n"
        "volume:          %.7f m^3\n"
        "speed:           %.7f m/s\n"
        "max timestep:    %.7f s\n"
        "min wavelength:  %.7f m\n"
        "max frequency:   %.2f hz\n"
        "damp:            %.2f\n\n",
        params.res.x, params.res.y, params.res.z,
        params.subGridRes.x, params.subGridRes.y, params.subGridRes.z,
        params.useSubGrid ? "True" : "False",
        params.step,
        dims[0], dims[1], dims[2],
        params.getVolume(),
        params.speed,
        params.getMaxTimestep(),
        params.getMinWavelength(),
        params.getMaxFrequency(),
        params.damp);

    std::vector<float> initial;
    initial.resize(params.res.x * params.res.y * params.res.z);
    for (auto& v : initial)
        v = Random::nextFloat(-0.1f, 0.1f);

    Wave3D wave(params, initial);
    float timestep = params.getMaxTimestep();
    uint32_t runs = 10;

    auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < runs; i++)
    {
        wave.increment(timestep);
    }
    auto duration = std::chrono::high_resolution_clock::now() - start;
    float ms = std::chrono::duration_cast<std::chrono::microseconds>(duration).count() / 1000.0f;

    uint64_t numPoints = params.res.x * params.res.y * params.res.z;

    float pointsPerSec = ((float)runs * (float)numPoints) / (ms / 1000.0f);

    std::cout << strFormat(
        "%u steps done in %.1f ms  -  %.1f million points / sec\n",
        runs,
        ms,
        pointsPerSec / 1000000.0f);

    std::cout << strFormat("sim time: %f s\n", wave.getTotalTime());

    return 0;
}
