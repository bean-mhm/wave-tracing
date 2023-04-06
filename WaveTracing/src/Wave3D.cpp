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
    return 2.0f * minStep;
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
    m_numValues = m_params.planarOrdering ? m_numPoints : (m_numPoints * 2);

    // Initial values
    if (initialValues.size() < 1)
    {
        m_valuesA.resize(m_numValues);
        for (auto& v : m_valuesA)
            v = 0.0f;
    }
    else if (initialValues.size() == m_numPoints)
    {
        if (m_params.planarOrdering)
        {
            m_valuesA = initialValues;
        }
        else
        {
            m_valuesA.resize(m_numValues);
            for (int i = 0; i < initialValues.size(); i++)
            {
                m_valuesA[i * 2] = initialValues[i];
                m_valuesA[i * 2 + 1] = initialValues[i];
            }
        }
    }
    else
    {
        throw std::exception("Invalid size of initial values.");
    }

    // Alternate buffer
    if (m_params.planarOrdering)
    {
        m_valuesB = m_valuesA;
    }

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
    const std::vector<float>& currValues = m_alternate ? m_valuesB : m_valuesA;
    std::vector<float>& prevValues = m_alternate ? m_valuesA : m_valuesB;

    // Alternate between the first and second compotents of each element (contagious ordering)
    const int currIndexOffset = m_alternate ? 0 : 1;
    const int prevIndexOffset = m_alternate ? 1 : 0;

    // Alternation indicator
    m_alternate = !m_alternate;

    // Eliminate repeated calculations in for loops

    const float dampMul = powf(m_params.damp, -timestep);
    const float accMul = timestep / powf(m_params.step, 2.0f);
    const float velMul = dampMul * timestep;

    const bool planar = m_params.planarOrdering;
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

                if (planar)
                {
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
                else
                {
                    index *= 2;

                    // Get the current value of this point
                    float curr = m_valuesA[index + currIndexOffset];

                    // Calculate the gradients

                    float gradZ =
                        (((z + 1 >= m_params.resZ) ? 0.0f : m_valuesA[index + strideZ + currIndexOffset]) - curr)
                        - (curr - ((z == 0) ? 0.0f : m_valuesA[index - strideZ + currIndexOffset]));

                    float gradY =
                        (((y + 1 >= m_params.resY) ? 0.0f : m_valuesA[index + strideY + currIndexOffset]) - curr)
                        - (curr - ((y == 0) ? 0.0f : m_valuesA[index - strideY + currIndexOffset]));

                    float gradX =
                        (((x + 1 >= m_params.resX) ? 0.0f : m_valuesA[index + 1 + currIndexOffset]) - curr)
                        - (curr - ((x == 0) ? 0.0f : m_valuesA[index - 1 + currIndexOffset]));

                    // Calculate the current velocity
                    float currVel = (curr - m_valuesA[index + prevIndexOffset]) / m_prevTimestep;

                    // Adjust the velocity
                    currVel += accMul * c2 * (gradX + gradY + gradZ);

                    // Store the new value in the other buffer
                    m_valuesA[index + prevIndexOffset] = curr + (currVel * velMul);
                }
            }
        }
    }

    m_prevTimestep = timestep;
    m_totalTime += timestep;
}

std::vector<float>& Wave3D::getValues()
{
    if (m_params.planarOrdering)
        return m_alternate ? m_valuesB : m_valuesA;
    else
        return m_valuesA;
}

std::vector<float>& Wave3D::getSpeedFactors()
{
    return m_speedFactors;
}

float Wave3D::getTotalTime()
{
    return m_totalTime;
}
