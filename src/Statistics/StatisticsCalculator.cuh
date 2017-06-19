#ifndef __STATISTICSCALCULATOR_CUH__
#define __STATISTICSCALCULATOR_CUH__

#include <assert.h>
#include <math.h>
#include "SATSolver/Configs.cuh"

template<class T>
class StatisticsCalculator
{
private:
    T *elements;
    int size;
    double average;
    double std_deviation;
    bool calculated;

public:

    StatisticsCalculator(T *elements, int size)
    {
        this->elements = elements;
        this->size = size;
        this->calculated = false;
    }

    void calculate()
    {
#ifdef USE_ASSERTIONS
        assert(!calculated);
#endif
        average = 0;
        std_deviation = 0;

        for (int i = 0; i < size; i++) {
            average += elements[i];
        }

        average = average / (double)size;

        for (int i = 0; i < size; i++) {
            std_deviation += (elements[i] - average) * (elements[i] - average);
        }

        std_deviation = std_deviation / size;

        std_deviation = sqrt(std_deviation);

        calculated = true;

    }

    double get_average()
    {
#ifdef USE_ASSERTIONS
        assert(calculated);
#endif
        return average;
    }

    double get_std_deviation()
    {
#ifdef USE_ASSERTIONS
        assert(calculated);
#endif
        return std_deviation;
    }


};

#endif /* __STATISTICSCALCULATOR_CUH__ */
