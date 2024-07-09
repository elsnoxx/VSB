#include "MinMax.h"

/**
 * @brief Retrieves the minimum temperature values.
 *
 * Searches for the minimum temperature values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the temperature data.
 * @return A vector containing the minimum temperature values.
 */
std::vector<double> getTemperaturMin(const std::map<int, Data>& dataMap) {
    std::vector<double> minValues;
    bool firstValue = true;
    double minValue;

    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double temperatur = dataObject.getTemperature();

        if (firstValue) {
            minValue = temperatur;
            firstValue = false;
            minValues.push_back(minValue);
        }
        else {
            if (temperatur < minValue) {
                minValue = temperatur;
                minValues.clear();
                minValues.push_back(minValue);
            }
            else if (temperatur == minValue) {
                minValues.push_back(minValue);
            }
        }
    }
    return minValues;
}

/**
 * @brief Retrieves the maximum temperature values.
 *
 * Searches for the maximum temperature values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the temperature data.
 * @return A vector containing the maximum temperature values.
 */
std::vector<double> getTemperaturMax(const std::map<int, Data>& dataMap) {
    std::vector<double> maxValues;
    bool firstValue = true;
    double maxValue;

    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double temperatur = dataObject.getTemperature();

        if (firstValue) {
            maxValue = temperatur;
            firstValue = false;
            maxValues.push_back(maxValue);
        }
        else {
            if (temperatur > maxValue) {
                maxValue = temperatur;
                maxValues.clear();
                maxValues.push_back(maxValue);
            }
            else if (temperatur == maxValue) {
                maxValues.push_back(maxValue);
            }
        }
    }
    return maxValues;
}

/**
 * @brief Retrieves the minimum humidity values.
 *
 * Searches for the minimum humidity values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the humidity data.
 * @return A vector containing the minimum humidity values.
 */
std::vector<double> getHumidityMin(const std::map<int, Data>& dataMap) {
    std::vector<double> maxValues;
    bool firstValue = true;
    double maxValue;

    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double humidity = dataObject.getHumidity();

        if (firstValue) {
            maxValue = humidity;
            firstValue = false;
            maxValues.push_back(maxValue);
        }
        else {
            if (humidity < maxValue) {
                maxValue = humidity;
                maxValues.clear();
                maxValues.push_back(maxValue);
            }
            else if (humidity == maxValue) {
                maxValues.push_back(maxValue);
            }
        }
    }

    return maxValues;
}

/**
 * @brief Retrieves the maximum humidity values.
 *
 * Searches for the maximum humidity values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the humidity data.
 * @return A vector containing the maximum humidity values.
 */
std::vector<double> getHumidityMax(const std::map<int, Data>& dataMap) {
    std::vector<double> maxValues;
    bool firstValue = true;
    double maxValue;

    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        double humidity = dataObject.getHumidity();

        if (firstValue) {
            maxValue = humidity;
            firstValue = false;
            maxValues.push_back(maxValue);
        }
        else {
            if (humidity > maxValue) {
                maxValue = humidity;
                maxValues.clear();
                maxValues.push_back(maxValue);
            }
            else if (humidity == maxValue) {
                maxValues.push_back(maxValue);
            }
        }
    }

    return maxValues;
}

/**
 * @brief Prints the maximum and minimum humidity and temperature values.
 *
 * This function prints out the maximum and minimum humidity and temperature values found in the data map.
 *
 * @param dataMap The unordered map containing the data.
 */
void printMaxAndMin(const std::map<int, Data>& dataMap) {
    std::vector<double> minTemperatur = getTemperaturMin(dataMap);
    std::cout << "Minimum temperature values: ";
    std::cout << minTemperatur.front();
    std::cout << std::endl;

    std::vector<double> maxTemperatur = getTemperaturMax(dataMap);
    std::cout << "Maximum temperature values: ";
    std::cout << maxTemperatur.front();
    std::cout << std::endl;

    std::vector<double> minHumidities = getHumidityMin(dataMap);
    std::cout << "Minimum humidity values: ";
    std::cout << minHumidities.front();
    std::cout << std::endl;


    std::vector<double> maxHumidities = getHumidityMax(dataMap);
    std::cout << "Maximum humidity values: ";
    std::cout << maxHumidities.front();
    std::cout << std::endl;
}