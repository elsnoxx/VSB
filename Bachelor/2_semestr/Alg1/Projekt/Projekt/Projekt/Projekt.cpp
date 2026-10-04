/**
* @file main.cpp
* @brief Main entry point of the program.
*
* This file contains the main function which serves as the entry point of the program.
* It includes headers for project-related classes and implements the program logic,
* including reading data from a file, processing data, generating histograms, and printing results.
*/
#include "Projekt.h"
#include "Task.h"
#include "Histogram.h"
#include "MinMax.h"



int main() {
    std::ifstream file("C:\\Users\\ficek\\Documents\\GitHub\\Alg1\\Projekt\\Projekt\\Projekt\\TestData\\IoTSensorData2.txt");

    if (!file.is_open()) {
        std::cerr << "Cannot open file." << std::endl;
        return 1;
    }

    int index = 0;
    std::map<int, Data> dataMap;
    std::string timestamp, hexValue;

    while (file >> timestamp >> hexValue) {
        Data dataObject(timestamp, hexValue);
        dataMap[index++] = dataObject; // Using index as key
    }

    file.close();

    // Function to sort input data
    quickSort(dataMap,0 , index - 1);
    std::cout << "size " << dataMap.size() << std::endl;

    // Function to remove duplicit in data
    removeDuplicates(dataMap);

    std::cout << "size " << dataMap.size() << std::endl;

    // Function to save data to file
    saveDataToFile(dataMap, "output.txt");
    std::cout << std::endl;

    // Function which will print all Max and Min Temperatures and Humidyt from input data
    printMaxAndMin(dataMap);
    std::cout << std::endl;

    // Build and print histogram of Temperatures
    drawHistogram(buildHistogramTemperature(dataMap, 10));
    std::cout << std::endl;
    std::cout << std::endl;
    // Build and print histogram of Humidity
    drawHistogram(buildHistogramHumidity(dataMap, 10));

    std::cout << std::endl;
    std::cout << std::endl;
    // Find Measurement on given timestamp
    findMeasurement(dataMap, "2024-02-06T09:43:00");
    findMeasurement(dataMap, "2023-03-10T22:00:00");

    return 0;
}