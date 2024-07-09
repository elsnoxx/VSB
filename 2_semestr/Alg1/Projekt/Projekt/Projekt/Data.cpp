#include "Data.h"

// Class constructor Data
Data::Data(const std::string& timestamp, const std::string& hexValue) {
    // Processing hexadecimal payload into numerical values
    processPayloadToNum(hexValue);
    // Splitting the timestamp into individual parts
    splitTimestamp(timestamp);
}

// Method to get formatted timestamp
std::string Data::getTimestamp() const {
    // Creating a formatted string for timestamp in ISO 8601 standard
    std::ostringstream oss;
    oss << year << "-" << std::setw(2) << std::setfill('0') << month << "-" << std::setw(2) << std::setfill('0') << day
        << "T" << std::setw(2) << std::setfill('0') << hour << ":" << std::setw(2) << std::setfill('0') << minute << ":" << std::setw(2) << std::setfill('0') << second;
    return oss.str();
}

// Method to get the value of B0
int Data::getB0() const {
    return B0;
}

// Method to get the value of B1
int Data::getB1() const {
    return B1;
}

// Method to get the value of B2
int Data::getB2() const {
    return B2;
}

// Method to get the value of B3
int Data::getB3() const {
    return B3;
}

// Method to get the value of B4
int Data::getB4() const {
    return B4;
}


// Method to get the year
int Data::getYear() const {
    return year;
}

// Method to get the month
int Data::getMonth() const {
    return month;
}

// Method to get the day
int Data::getDay() const {
    return day;
}

// Method to get the hour
int Data::getHour() const {
    return hour;
}

// Method to get the minute
int Data::getMinute() const {
    return minute;
}

// Method to get the second
int Data::getSecond() const {
    return second;
}

// Method for processing hexadecimal payload into numerical values
void Data::processPayloadToNum(const std::string& hexValue) {
    // Convert hexadecimal strings to integers
    B0 = hexToInt(hexValue.substr(0, 2));
    B1 = hexToInt(hexValue.substr(2, 2));
    B2 = hexToInt(hexValue.substr(4, 2));
    B3 = hexToInt(hexValue.substr(6, 2));
    B4 = hexToInt(hexValue.substr(8, 2));
}

// Method for splitting the timestamp into individual parts
void Data::splitTimestamp(const std::string& timestamp) {
    std::istringstream ss(timestamp);
    char delimiter;
    // Parsing the timestamp
    ss >> year >> delimiter >> month >> delimiter >> day >> delimiter >> hour >> delimiter >> minute >> delimiter >> second;
}

// Helper method to convert hexadecimal string to integer
std::int16_t Data::hexToInt(const std::string& hexStr) const {
    std::stringstream ss;
    ss << std::hex << hexStr;
    std::int16_t intValue;
    // Converting hexadecimal string to integer
    ss >> intValue;
    return intValue;
}

// Helper method to convert integer to hexadecimal string
std::string Data::intToHex(std::int16_t intValue) const {
    const std::string hexChars = "0123456789ABCDEF";
    std::string result;

    // Vytvoření hexadecimálního řetězce ručně
    result.push_back(hexChars[(intValue >> 4) & 0xF]);
    result.push_back(hexChars[intValue & 0xF]);

    return result;
}

// Method to get temperature from data
std::double_t  Data::getTemperature() const {
    // Calculating temperature based on values B1 and B2
    return (0.1 * (256 * B1 + B2));
}

// Method to get humidity from data
std::double_t  Data::getHumidity() const {
    // Calculating humidity based on values B3 and B4
    return (0.1 * (256 * B3 + B4));
}

// Method to get voltage from data
std::double_t  Data::getVoltage() const {
    // Calculating voltage based on value B0
    return (30 * getB0());
}

// Method to get payload
std::string Data::getPayload() const {
    // Convert integer values B0-B4 back to hexadecimal strings
    std::string payload = intToHex(B0) + intToHex(B1) + intToHex(B2) + intToHex(B3) + intToHex(B4);
    return payload;
}
