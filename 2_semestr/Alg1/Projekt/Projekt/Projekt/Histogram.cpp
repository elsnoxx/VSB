#include "Histogram.h"

/**
 * @brief Builds a histogram for temperature data.
 *
 * This function calculates the frequency of temperature values falling into specified intervals.
 *
 * @param dataMap The unordered map containing the temperature data.
 * @param numIntervals The number of intervals to divide the temperature range into.
 * @return A vector of HistogramInterval objects representing the histogram.
 */
std::vector<HistogramInterval> buildHistogramTemperature(const std::map<int, Data>& dataMap, int numIntervals) {
    // Získání rozsahu hodnot teploty
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::min();
    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double temperature = dataObject.getTemperature();
        if (temperature < minVal) {
            minVal = temperature;
        }
        if (temperature > maxVal) {
            maxVal = temperature;
        }
    }

    // Rozdìlení intervalù
    double intervalSize = (maxVal - minVal) / numIntervals;
    std::vector<HistogramInterval> histogram;
    for (int i = 0; i < numIntervals; ++i) {
        double lowerBound = minVal + i * intervalSize;
        double upperBound = minVal + (i + 1) * intervalSize;
        histogram.push_back({ lowerBound, upperBound, 0 });
    }

    // Výpoèet èetností
    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double temperature = dataObject.getTemperature();
        for (auto& interval : histogram) {
            if (temperature >= interval.lowerBound && temperature < interval.upperBound) {
                interval.frequency++;
                break;
            }
        }
    }

    return histogram;
}

/**
 * @brief Builds a histogram for humidity data.
 *
 * This function calculates the frequency of humidity values falling into specified intervals.
 *
 * @param dataMap The unordered map containing the humidity data.
 * @param numIntervals The number of intervals to divide the humidity range into.
 * @return A vector of HistogramInterval objects representing the histogram.
 */
std::vector<HistogramInterval> buildHistogramHumidity(const std::map<int, Data>& dataMap, int numIntervals) {
    // Získání rozsahu hodnot vlhkosti
    double minVal = std::numeric_limits<double>::max();
    double maxVal = std::numeric_limits<double>::min();
    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double humidity = dataObject.getHumidity();
        if (humidity < minVal) {
            minVal = humidity;
        }
        if (humidity > maxVal) {
            maxVal = humidity;
        }
    }

    // Rozdìlení intervalù
    double intervalSize = (maxVal - minVal) / numIntervals;
    std::vector<HistogramInterval> histogram;
    for (int i = 0; i < numIntervals; ++i) {
        double lowerBound = minVal + i * intervalSize;
        double upperBound = minVal + (i + 1) * intervalSize;
        histogram.push_back({ lowerBound, upperBound, 0 });
    }

    // Výpoèet èetností
    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double humidity = dataObject.getHumidity();
        for (auto& interval : histogram) {
            if (humidity >= interval.lowerBound && humidity < interval.upperBound) {
                interval.frequency++;
                break;
            }
        }
    }

    return histogram;
}

/**
 * @brief Draws a histogram based on the provided histogram intervals.
 *
 * This function draws a histogram based on the provided histogram intervals.
 *
 * @param histogram The vector of HistogramInterval objects representing the histogram.
 */
void drawHistogram(const std::vector<HistogramInterval>& histogram) {
    int maxFrequency = 0;
    for (const auto& interval : histogram) {
        if (interval.frequency > maxFrequency) {
            maxFrequency = interval.frequency;
        }
    }

    // Vypsání histogramu jako tabulky
    std::cout << "Histogram" << std::endl;
    for (const auto& interval : histogram) {
        std::cout << std::fixed << std::setprecision(2) << std::setw(5) << interval.lowerBound << " - "
            << std::setw(5) << interval.upperBound << " | ";
        // Vypsání hvìzdièek odpovídajících frekvenci v daném intervalu
        int numStars = static_cast<int>(40.0 * interval.frequency / maxFrequency); // Normalizace na maximálnì 40 hvìzdièek
        for (int i = 0; i < numStars; ++i) {
            std::cout << "*";
        }
        std::cout << std::endl;
    }
}
