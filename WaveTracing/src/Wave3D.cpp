#include "Wave3D.h"

#include <omp.h>

#define WAVE3D_VEC

#ifdef WAVE3D_VEC

#include <xmmintrin.h>

/*

https://stackoverflow.com/a/11228864/18049911

<mmintrin.h>  MMX
<xmmintrin.h> SSE
<emmintrin.h> SSE2
<pmmintrin.h> SSE3
<tmmintrin.h> SSSE3
<smmintrin.h> SSE4.1
<nmmintrin.h> SSE4.2
<ammintrin.h> SSE4A
<wmmintrin.h> AES
<immintrin.h> AVX, AVX2, FMA

*/

#endif

float WaveParams::getMaxTimestep()
{
    return step / (sqrtf(3.0f) * speed);
}

float WaveParams::getMinWavelength()
{
    float minStep = sqrtf(3.0f) * step;
    return 2.0f * minStep;
}

float WaveParams::getMaxFrequency()
{
    return speed / getMinWavelength();
}

std::array<float, 3> WaveParams::getDimensions()
{
    float dimX = (float)(res.x - 1) * step;
    float dimY = (float)(res.y - 1) * step;
    float dimZ = (float)(res.z - 1) * step;
    return { dimX, dimY, dimZ };
}

float WaveParams::getVolume()
{
    std::array<float, 3> dims = getDimensions();
    return dims[0] * dims[1] * dims[2];
}

Wave3D::Wave3D(const WaveParams& params, const std::vector<float>& initialValues, const std::vector<float>& speedFactors)
    : m_params(params)
{
    // Verify the parameters

    if (m_params.res.x < 1 || m_params.res.y < 1 || m_params.res.z < 1)
        throw std::exception("A resolution of at least 1x1x1 is required.");

    if (m_params.step <= 0.0f)
        throw std::exception("Step must be a positive real number.");

    if (m_params.damp < 1.0f)
        throw std::exception("Damp must be a real number larger than or equal to 1.");

    if (m_params.useSubGrid)
    {
        bool validSubRes =
            (m_params.subGridRes.x < m_params.res.x)
            && (m_params.subGridRes.y < m_params.res.y)
            && (m_params.subGridRes.z < m_params.res.z);

        validSubRes &=
            (m_params.subGridRes.x > 1)
            && (m_params.subGridRes.y > 1)
            && (m_params.subGridRes.z > 1);

        if (!validSubRes)
            m_params.useSubGrid = false;
    }

    // Total number of points to simulate
    m_numPoints = m_params.res.x * m_params.res.y * m_params.res.z;

    // Initial values
    if (initialValues.size() < 1)
    {
        m_valuesA.resize(m_numPoints);
        for (auto& v : m_valuesA)
            v = 0.0f;
    }
    else if (initialValues.size() != m_numPoints)
    {
        throw std::exception("Invalid size of initial values.");
    }
    else
    {
        m_valuesA = initialValues;
    }
    m_valuesB = m_valuesA;

    // Initial speed factors
    if (speedFactors.size() < 1)
    {
        m_speedFactors.resize(m_numPoints);
        for (auto& v : m_speedFactors)
            v = 1.0f;
    }
    else if (speedFactors.size() != m_numPoints)
    {
        throw std::exception("Invalid size of initial speed factors.");
    }
    else
    {
        m_speedFactors = speedFactors;
    }
}

void Wave3D::increment(float timestep)
{
    if (timestep == 0.0f)
        return;

    if (m_lastTimestep == 0.0f)
        m_lastTimestep = timestep;

    // Alternate between m_valuesA and m_valuesB
    std::vector<float>& currValues = m_alternate ? m_valuesB : m_valuesA;
    std::vector<float>& lastValues = m_alternate ? m_valuesA : m_valuesB;
    m_alternate = !m_alternate;

    // Eliminate repeated calculations in for loops
    float dampMul = powf(m_params.damp, -timestep);
    float accMul = timestep / powf(m_params.step, 2.0f);
    float velMul = dampMul * timestep;

    uint32_t strideZ = m_params.res.x * m_params.res.y;

#pragma omp parallel for
    for (int z = 0; z < m_params.res.z; z++)
    {
        for (int y = 0; y < m_params.res.y; y++)
        {
            for (int x = 0; x < m_params.res.x; x++)
            {
                // uint32_t currIndex = index3D(x, y, z, m_params.res.x, m_params.res.Y);
                uint32_t currIndex = (z * strideZ) + (y * m_params.res.x) + x;

                // Skip this point if the speed factor is 0
                if (m_speedFactors[currIndex] == 0.0f)
                    continue;

                // Calculate speed^2
                float c2 = m_params.speed * m_speedFactors[currIndex];
                c2 *= c2;

                // Get the current value of this point
                float curr = currValues[currIndex];

                // Calculate the gradients

                float gradX =
                    (((x + 1 >= m_params.res.x) ? 0.0f : currValues[currIndex + 1]) - curr)
                    - (curr - ((x == 0) ? 0.0f : currValues[currIndex - 1]));

                float gradY =
                    (((y + 1 >= m_params.res.y) ? 0.0f : currValues[currIndex + m_params.res.x]) - curr)
                    - (curr - ((y == 0) ? 0.0f : currValues[currIndex - m_params.res.x]));

                float gradZ =
                    (((z + 1 >= m_params.res.z) ? 0.0f : currValues[currIndex + strideZ]) - curr)
                    - (curr - ((z == 0) ? 0.0f : currValues[currIndex - strideZ]));

                // Calculate the current velocity
                float currVel = (curr - lastValues[currIndex]) / m_lastTimestep;

                // Adjust the velocity
                currVel += accMul * c2 * (gradX + gradY + gradZ);

                // Store the new value in the other buffer
                lastValues[currIndex] = curr + (currVel * velMul);
            }
        }
    }

    m_lastTimestep = timestep;
    m_totalTime += timestep;
}

std::vector<float>& Wave3D::getValues()
{
    return m_alternate ? m_valuesB : m_valuesA;
}

std::vector<float>& Wave3D::getSpeedFactors()
{
    return m_speedFactors;
}

float Wave3D::getTotalTime()
{
    return m_totalTime;
}
