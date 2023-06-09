#include "Wave3D.h"

#include <omp.h>

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
    return 8.0f * minStep;
}

float WaveParams::getMaxFrequency()
{
    return speed / getMinWavelength();
}

std::array<float, 3> WaveParams::getDimensions()
{
    float dimX = (float)(resX - 1) * step;
    float dimY = (float)(resY - 1) * step;
    float dimZ = (float)(resZ - 1) * step;
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

    if (m_params.resX < 1 || m_params.resY < 1 || m_params.resZ < 1)
        throw std::exception("A resolution of at least 1x1x1 is required.");

    if (m_params.step <= 0.0f)
        throw std::exception("Step must be a positive real number.");

    if (m_params.damp < 1.0f)
        throw std::exception("Damp must be a real number larger than or equal to 1.");

    // Total number of points to simulate
    m_numPoints = m_params.resX * m_params.resY * m_params.resZ;

    // Initial values
    if (initialValues.size() < 1)
    {
        m_valuesA.resize(m_numValues);
        for (auto& v : m_valuesA)
            v = 0.0f;
    }
    else if (initialValues.size() == m_numPoints)
    {
        m_valuesA = initialValues;
    }
    else
    {
        throw std::exception("Invalid size of initial values.");
    }

    // Alternate buffer
    m_valuesB = m_valuesA;

    // Initial speed factors
    if (speedFactors.size() < 1)
    {
        m_speedFactors.resize(m_numPoints);
        for (auto& v : m_speedFactors)
            v = 1.0f;
    }
    else if (speedFactors.size() == m_numPoints)
    {
        m_speedFactors = speedFactors;
    }
    else
    {
        throw std::exception("Invalid size of initial speed factors.");
    }
}

void Wave3D::increment(float timestep)
{
    if (timestep == 0.0f)
        return;

    if (m_prevTimestep == 0.0f)
        m_prevTimestep = timestep;

    // Alternate between m_valuesA and m_valuesB (planar ordering)
    const auto& currValues = m_alternate ? m_valuesB : m_valuesA;
    auto& prevValues = m_alternate ? m_valuesA : m_valuesB;
    m_alternate = !m_alternate;

    // Eliminate repeated calculations in for loops

    const float dampMul = powf(m_params.damp, -timestep);
    const float accMul = timestep / powf(m_params.step, 2.0f);
    const float velMul = dampMul * timestep;

    const int strideY = m_params.resX;
    const int strideZ = m_params.resX * m_params.resY;

#pragma omp parallel for
    for (int z = 0; z < m_params.resZ; z++)
    {
        for (int y = 0; y < m_params.resY; y++)
        {
            for (int x = 0; x < m_params.resX; x++)
            {
                uint32_t index = (z * strideZ) + (y * strideY) + x;

                // Calculate speed^2
                float c2 = m_params.speed * m_speedFactors[index];
                c2 *= c2;

                // Get the current value of this point
                float curr = currValues[index];

                // Calculate the gradients

                float gradZ =
                    (((z + 1 >= m_params.resZ) ? 0.0f : currValues[index + strideZ]) - curr)
                    - (curr - ((z == 0) ? 0.0f : currValues[index - strideZ]));

                float gradY =
                    (((y + 1 >= m_params.resY) ? 0.0f : currValues[index + strideY]) - curr)
                    - (curr - ((y == 0) ? 0.0f : currValues[index - strideY]));

                float gradX =
                    (((x + 1 >= m_params.resX) ? 0.0f : currValues[index + 1]) - curr)
                    - (curr - ((x == 0) ? 0.0f : currValues[index - 1]));

                // Calculate the current velocity
                float currVel = (curr - prevValues[index]) / m_prevTimestep;

                // Adjust the velocity
                currVel += accMul * c2 * (gradX + gradY + gradZ);

                // Store the new value in the other buffer
                prevValues[index] = curr + (currVel * velMul);
            }
        }
    }

    m_prevTimestep = timestep;
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
