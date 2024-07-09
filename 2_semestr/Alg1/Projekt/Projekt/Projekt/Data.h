#pragma once

#include "Projekt.h"

/**
 * @brief Class representing a data entry.
 *
 * This class stores various data fields including timestamps, sensor readings, and other payload information.
 */
class Data {
private:
    int B0; ///< Sensor reading or payload information.
    int B1; ///< Sensor reading or payload information.
    int B2; ///< Sensor reading or payload information.
    int B3; ///< Sensor reading or payload information.
    int B4; ///< Sensor reading or payload information.
    int year; ///< Year component of the timestamp.
    int month; ///< Month component of the timestamp.
    int day; ///< Day component of the timestamp.
    int hour; ///< Hour component of the timestamp.
    int minute; ///< Minute component of the timestamp.
    int second; ///< Second component of the timestamp.

public:
    /**
     * @brief Default constructor.
     *
     * Initializes data fields with default values.
     */
    Data() : year(0), month(0), day(0), hour(0), minute(0), second(0), B0(0), B1(0), B2(0), B3(0), B4(0) {}

    /**
     * @brief Constructor with timestamp and payload.
     *
     * Initializes data fields with the provided timestamp and payload.
     *
     * @param timestamp The timestamp string.
     * @param hexValue The hexadecimal payload.
     */
    Data(const std::string& timestamp, const std::string& hexValue);

    /**
     * @brief Gets the timestamp string.
     *
     * @return The timestamp string.
     */
    std::string getTimestamp() const;

    /**
     * @brief Splits the timestamp string into its components.
     *
     * @param timestamp The timestamp string to split.
     */
    void splitTimestamp(const std::string& timestamp);

    /**
     * @brief Gets sensor reading B0.
     *
     * @return Sensor reading B0.
     */
    int getB0() const;

    /**
     * @brief Gets sensor reading B1.
     *
     * @return Sensor reading B1.
     */
    int getB1() const;

    /**
     * @brief Gets sensor reading B2.
     *
     * @return Sensor reading B2.
     */
    int getB2() const;

    /**
     * @brief Gets sensor reading B3.
     *
     * @return Sensor reading B3.
     */
    int getB3() const;

    /**
     * @brief Gets sensor reading B4.
     *
     * @return Sensor reading B4.
     */
    int getB4() const;
    
    /**
     * @brief Get the year.
     *
     * @return The year.
     */
    int getYear() const;

    /**
     * @brief Get the month.
     *
     * @return The month.
     */
    int getMonth() const;

    /**
     * @brief Get the day.
     *
     * @return The day.
     */
    int getDay() const;

    /**
     * @brief Get the hour.
     *
     * @return The hour.
     */
    int getHour() const;

    /**
     * @brief Get the minute.
     *
     * @return The minute.
     */
    int getMinute() const;

    /**
     * @brief Get the second.
     *
     * @return The second.
     */
    int getSecond() const;


    /**
     * @brief Processes the hexadecimal payload to extract numerical values.
     *
     * Converts the hexadecimal payload to numerical values for temperature, humidity, and voltage.
     *
     * @param hexValue The hexadecimal payload.
     */
    void processPayloadToNum(const std::string& hexValue);

    /**
     * @brief Converts hexadecimal string to integer.
     *
     * @param hexStr The hexadecimal string to convert.
     * @return The integer value.
     */
    std::int16_t hexToInt(const std::string& hexStr) const;

    std::string intToHex(std::int16_t intValue) const;

    /**
     * @brief Gets the temperature value.
     *
     * @return The temperature value.
     */
    std::double_t getTemperature() const;

    /**
     * @brief Gets the humidity value.
     *
     * @return The humidity value.
     */
    std::double_t getHumidity() const;

    /**
     * @brief Gets the voltage value.
     *
     * @return The voltage value.
     */
    std::double_t getVoltage() const;

    /**
     * @brief Gets the payload string.
     *
     * @return The payload string.
     */
    std::string getPayload() const;
};
