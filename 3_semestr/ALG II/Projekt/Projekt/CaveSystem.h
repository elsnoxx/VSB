#ifndef CAVESYSTEM_H
#define CAVESYSTEM_H

#include <string>
#include <map>
#include "Cave.h"

/**
 * @class CaveSystem
 * @brief Represents a system of caves and their connections.
 *
 * This class allows loading caves and their connections from a file.
 */
class CaveSystem {
private:
    /**
     * @brief A map of cave names to Cave objects.
     */
    std::map<std::string, Cave> caves;

public:
    /**
     * @brief Loads cave connections from a file.
     *
     * The file should contain lines in the format "cave1-cave2",
     * indicating a bidirectional connection between two caves.
     *
     * @param filename The path to the file containing cave connections.
     * @throw std::runtime_error If the file cannot be opened.
     */
    void loadFromFile(const std::string& filename);

    /**
     * @brief Gets the map of caves.
     *
     * @return A constant reference to the map of caves.
     */
    const std::map<std::string, Cave>& getCaves() const;
};

#endif // CAVESYSTEM_H
