#include "Task.h"
#include <iostream>
#include <fstream>

#include "Data.h"

/**
 * @brief Saves the data to a file.
 *
 * This function saves the data stored in the provided data map to a file with the specified filename.
 * Each line in the file contains the timestamp and payload of the data.
 *
 * @param dataMap The unordered map containing the data to be saved.
 * @param filename The name of the file to save the data to.
 */
void saveDataToFile(const std::map<int, Data>& dataMap, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (outputFile.is_open()) {
        for (const auto& pair : dataMap) {
            const Data& dataObject = pair.second;
            outputFile << dataObject.getTimestamp() << " " << dataObject.getPayload() << std::endl;
        }
        outputFile.close();
        std::cout << "The data was write into " << filename << "\n";
    }
    else {
        std::cerr << "Error opening file " << filename << "\n";
    }
}

/**
 * @brief Finds and returns the measured values for a given timestamp.
 *
 * Searches for and returns the measured values for a specific timestamp in the data map.
 *
 * @param dataMap The unordered map containing the data.
 * @param timestamp The timestamp for which measurements are requested.
 */
void findMeasurement(const std::map<int, Data>& dataMap, const std::string& timestamp) {
    int found = 0;
    for (const auto& pair : dataMap) {
        const Data& dataObject = pair.second;
        if (dataObject.getTimestamp() == timestamp)
        {
            std::cout << "For this timestamp: " << timestamp << " Temperature is: " << dataObject.getTemperature() << " and is Humidity: " << dataObject.getHumidity() << std::endl;
            found = 1;
        }
    }
    if (found == 0)
    {
        std::cout << "Not found" << std::endl;
    }
    
}


/**
 * @brief Compares two timestamps.
 *
 * Compares two timestamps for sorting purposes.
 *
 * @param data1 The first timestamp to compare.
 * @param data2 The second timestamp to compare.
 * @return True if data1 is earlier than data2, false otherwise.
 */
bool compareTimestamp(const Data& data1, const Data& data2) {
    if (data1.getYear() != data2.getYear())
        return data1.getYear() < data2.getYear();

    if (data1.getMonth() != data2.getMonth())
        return data1.getMonth() < data2.getMonth();

    if (data1.getDay() != data2.getDay())
        return data1.getDay() < data2.getDay();

    if (data1.getHour() != data2.getHour())
        return data1.getHour() < data2.getHour();

    if (data1.getMinute() != data2.getMinute())
        return data1.getMinute() < data2.getMinute();

    if (data1.getSecond() != data2.getSecond())
        return data1.getSecond() < data2.getSecond();

    return false;
}

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
int partition(std::map<int, Data>& dataMap, int low, int high) {
    Data pivot = dataMap[(low + high) / 2];
    int i = low - 1;
    int j = high + 1;

    while (true) {
        do {
            ++i;
        } while (compareTimestamp(dataMap[i], pivot));

        do {
            --j;
        } while (compareTimestamp(pivot, dataMap[j]));

        if (i >= j) {
            return j;
        }

        std::swap(dataMap[i], dataMap[j]);
    }
}

/**
 * @brief Implements the quicksort algorithm.
 *
 * Sorts the data map using the quicksort algorithm.
 *
 * @param dataMap The unordered map containing the data to be sorted.
 * @param low The starting index of the sort.
 * @param high The ending index of the sort.
 */
void quickSort(std::map<int, Data>& dataMap, int low, int high) {
    if (low < high) {
        int pi = partition(dataMap, low, high);
        quickSort(dataMap, low, pi);
        quickSort(dataMap, pi + 1, high);
    }
}


/**
 * @brief Removes duplicate entries from the data map based on timestamp.
 *
 * This function iterates through the provided data map and removes duplicate entries based on their timestamp.
 * Only the first occurrence of each unique timestamp is preserved, while all subsequent occurrences are removed.
 *
 * @param dataMap The unordered map containing the data to be processed.
 */
void removeDuplicates(std::map<int, Data>& dataMap) {
    std::map<std::string, bool> seenTimestamps;

    auto it = dataMap.begin();
    while (it != dataMap.end()) {
        const std::string& currentTimestamp = it->second.getTimestamp();
        if (seenTimestamps.find(currentTimestamp) != seenTimestamps.end()) {
            it = dataMap.erase(it);
        }
        else {
            seenTimestamps[currentTimestamp] = true;
            ++it;
        }
    }
}