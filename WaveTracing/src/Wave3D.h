#pragma once

#include <cmath>
#include <array>
#include <vector>
#include <cstdint>

struct WaveParams
{
    uint32_t resX = 1;
    uint32_t resY = 1;
    uint32_t resZ = 1;

    float step = 1.0f;
    float speed = 1.0f;
    float damp = 1.0f;

    float getMaxTimestep();
    float getMinWavelength();
    float getMaxFrequency();
    std::array<float, 3> getDimensions();
    float getVolume();
};

class Wave3D
{
private:
    WaveParams m_params;

    std::vector<float> m_valuesA;
    std::vector<float> m_valuesB;
    bool m_alternate = false;

    std::vector<float> m_speedFactors;

    uint32_t m_numPoints = 0;
    float m_lastTimestep = 0.0f;
    float m_totalTime = 0.0f;

public:
    Wave3D() = delete;
    Wave3D(const WaveParams& params, const std::vector<float>& initialValues = {}, const std::vector<float>& speedFactors = {});

    void increment(float timestep);

    std::vector<float>& getValues();
    std::vector<float>& getSpeedFactors();

    float getTotalTime();

};

inline int index3D(int x, int y, int z, int strideY, int strideZ)
{
    return (z * strideZ) + (y * strideY) + x;
}

inline bool inBounds3D(int x, int y, int z, int resX, int resY, int resZ)
{
    return
        (x < resX)
        && (y < resY)
        && (z < resZ)
        && (x >= 0)
        && (y >= 0)
        && (z >= 0);
}
