#pragma once

#include "Projekt.h"
#include "Data.h"

/**
 * @brief Retrieves the minimum temperature values.
 *
 * Searches for the minimum temperature values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the temperature data.
 * @return A vector containing the minimum temperature values.
 */
std::vector<double> getTemperatureMin(const std::map<int, Data>& dataMap);

/**
 * @brief Retrieves the maximum temperature values.
 *
 * Searches for the maximum temperature values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the temperature data.
 * @return A vector containing the maximum temperature values.
 */
std::vector<double> getTemperatureMax(const std::map<int, Data>& dataMap);

/**
 * @brief Retrieves the minimum humidity values.
 *
 * Searches for the minimum humidity values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the humidity data.
 * @return A vector containing the minimum humidity values.
 */
std::vector<double> getHumidityMin(const std::map<int, Data>& dataMap);

/**
 * @brief Retrieves the maximum humidity values.
 *
 * Searches for the maximum humidity values in the data map and returns them as a vector.
 *
 * @param dataMap The unordered map containing the humidity data.
 * @return A vector containing the maximum humidity values.
 */
std::vector<double> getHumidityMax(const std::map<int, Data>& dataMap);

/**
 * @brief Prints the maximum and minimum humidity and temperature values.
 *
 * This function prints out the maximum and minimum humidity and temperature values found in the data map.
 *
 * @param dataMap The unordered map containing the data.
 */
void printMaxAndMin(const std::map<int, Data>& dataMap);
