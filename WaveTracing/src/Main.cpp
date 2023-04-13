#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <chrono>
#include <thread>

#include "Wave3D.h"

#include "Utils/NumberHelpers.h"
#include "Utils/Random.h"
#include "Utils/Misc.h"

std::string strFromMeters(float meters, const std::string& format = "%.3f")
{
    int order = floorf(log10f(meters));

    if (order < -6)
    {
        return strFormat(format + " nm", meters * powf(10.0f, 9.0f));
    }
    else if (0 && (order < -3))
    {
        return strFormat(format + " microns", meters * powf(10.0f, 6.0f));
    }
    else if (order < -2)
    {
        return strFormat(format + " mm", meters * powf(10.0f, 3.0f));
    }
    else if (order < 0)
    {
        return strFormat(format + " cm", meters * powf(10.0f, 2.0f));
    }
    else if (order < 3)
    {
        return strFormat(format + " m", meters);
    }
    else
    {
        return strFormat(format + " km", meters * powf(10.0f, -3.0f));
    }
}

std::string strFromSeconds(float seconds, const std::string& format = "%.3f")
{
    int order = floorf(log10f(seconds));

    if (order < -6)
    {
        return strFormat(format + " ns", seconds * powf(10.0f, 9.0f));
    }
    else if (0 && (order < -3))
    {
        return strFormat(format + " microsec", seconds * powf(10.0f, 6.0f));
    }
    else if (order < 0)
    {
        return strFormat(format + " ms", seconds * powf(10.0f, 3.0f));
    }

    if (seconds < 60.0f)
    {
        return strFormat(format + " s", seconds);
    }
    else
    {
        uint64_t intSec = (int)floorf(seconds);
        uint64_t intHr = intSec / 3600;
        uint64_t intMin = (intSec / 60) % 60;
        intSec %= 60;

        if (intHr > 0)
        {
            return strFormat("%uh %um %us", intHr, intMin, intSec);
        }
        else
        {
            return strFormat("%um %us", intMin, intSec);
        }
    }
}

int main()
{
    WaveParams params;
    params.resX = 501;
    params.resY = 501;
    params.resZ = 501;
    params.step = 0.0025f;
    params.speed = 100.0f;
    params.damp = 1.0f;

    uint64_t numPoints = params.resX * params.resY * params.resZ;
    auto dims = params.getDimensions();

    float timestep = 0.95f * params.getMaxTimestep();
    uint32_t runs = 20;

    std::cout << strFormat(
        "resolution:      %u x %u x %u\n"
        "step size:       %s\n"
        "dimensions:      %s x %s x %s\n"
        "volume:          %.7f m^3\n"
        "speed:           %s/s\n"
        "max timestep:    %s\n"
        "min wavelength:  %s\n"
        "max frequency:   %.2f Hz\n"
        "damp:            %.2f\n\n",
        params.resX, params.resY, params.resZ,
        strFromMeters(params.step),
        strFromMeters(dims[0]), strFromMeters(dims[1]), strFromMeters(dims[2]),
        params.getVolume(),
        strFromMeters(params.speed),
        strFromSeconds(params.getMaxTimestep()),
        strFromMeters(params.getMinWavelength()),
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
