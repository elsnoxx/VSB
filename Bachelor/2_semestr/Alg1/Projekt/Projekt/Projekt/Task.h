#pragma once

#include <map>
#include <vector>
#include "Data.h"


/**
 * @brief Saves data to a file.
 *
 * Creates a new file and writes all data into it.
 *
 * @param dataMap The unordered map containing the data to be saved.
 * @param filename The name of the file to save the data to.
 */
void saveDataToFile(const std::map<int, Data>& dataMap, const std::string& filename);


/**
 * @brief Finds and returns the measured values for a given timestamp.
 *
 * Searches for and returns the measured values for a specific timestamp in the data map.
 *
 * @param dataMap The unordered map containing the data.
 * @param timestamp The timestamp for which measurements are requested.
 */
void findMeasurement(const std::map<int, Data>& dataMap, const std::string& timestamp);

/**
 * @brief Compares two timestamps.
 *
 * Compares two timestamps for sorting purposes.
 *
 * @param data1 The first timestamp to compare.
 * @param data2 The second timestamp to compare.
 * @return True if data1 is earlier than data2, false otherwise.
 */
bool compareTimestamp(const Data& data1, const Data& data2);

/**
 * @brief Partitions the data for quicksort.
 *
 * Partitions the data map for the quicksort algorithm.
 *
 * @param dataMap The unordered map containing the data to be partitioned.
 * @param low The starting index of the partition.
 * @param high The ending index of the partition.
 * @return The partition index.
 */
int partition(std::map<int, Data>& dataMap, int low, int high);

/**
 * @brief Implements the quicksort algorithm.
 *
 * Sorts the data map using the quicksort algorithm.
 *
 * @param dataMap The unordered map containing the data to be sorted.
 * @param low The starting index of the sort.
 * @param high The ending index of the sort.
 */
void quickSort(std::map<int, Data>& dataMap, int low, int high);


/**
 * @brief Removes duplicate entries from the data map based on timestamp.
 *
 * This function iterates through the provided data map and removes duplicate entries based on their timestamp.
 * Only the first occurrence of each unique timestamp is preserved, while all subsequent occurrences are removed.
 *
 * @param dataMap The unordered map containing the data to be processed.
 */
void removeDuplicates(std::map<int, Data>& dataMap);
