#pragma once

#include "Data.h"

/**
 * @brief Struct representing an interval in a histogram.
 *
 * This struct defines an interval in a histogram, containing lower and upper bounds
 * and the frequency of values falling within this interval.
 */
struct HistogramInterval {
    double lowerBound; ///< The lower bound of the interval.
    double upperBound; ///< The upper bound of the interval.
    int frequency;     ///< The frequency of values falling within this interval.
};

/**
 * @brief Builds a histogram for temperature data.
 *
 * This function builds a histogram for temperature data based on the provided data map and the specified number of intervals.
 *
 * @param dataMap The unordered map containing the temperature data.
 * @param numIntervals The number of intervals to divide the temperature range into.
 */
std::vector<HistogramInterval> buildHistogramTemperature(const std::map<int, Data>& dataMap, int numIntervals);

/**
 * @brief Builds a histogram for humidity data.
 *
 * This function builds a histogram for humidity data based on the provided data map and the specified number of intervals.
 *
 * @param dataMap The unordered map containing the humidity data.
 * @param numIntervals The number of intervals to divide the humidity range into.
 */
std::vector<HistogramInterval> buildHistogramHumidity(const std::map<int, Data>& dataMap, int numIntervals);

/**
 * @brief Draws a histogram based on the provided histogram intervals.
 *
 * This function draws a histogram based on the provided histogram intervals.
 *
 * @param histogram The vector of HistogramInterval objects representing the histogram.
 */
void drawHistogram(const std::vector<HistogramInterval>& histogram);