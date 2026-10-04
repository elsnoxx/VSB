#ifndef CAVE_H
#define CAVE_H

#include <string>
#include <vector>

/**
 * @class Cave
 * @brief Represents a single cave and its connections to other caves.
 *
 * This class stores the name of a cave and a list of its neighboring caves.
 */
class Cave {
private:
    /**
     * @brief The name of the cave.
     */
    std::string name;

    /**
     * @brief A list of names of neighboring caves.
     */
    std::vector<std::string> passages;

public:
    /**
     * @brief Default constructor.
     */
    Cave() = default;

    /**
     * @brief Constructs a cave with a given name.
     *
     * @param name The name of the cave.
     */
    Cave(const std::string& name);

    /**
     * @brief Adds a passage to a neighboring cave.
     *
     * @param neighbor The name of the neighboring cave.
     */
    void addPassage(const std::string& neighbor);

    /**
     * @brief Gets the name of the cave.
     *
     * @return The name of the cave.
     */
    const std::string& getName() const;

    /**
     * @brief Gets the list of passages.
     *
     * @return A constant reference to the vector of neighboring caves.
     */
    const std::vector<std::string>& getPassages() const;
};

#endif // CAVE_H
