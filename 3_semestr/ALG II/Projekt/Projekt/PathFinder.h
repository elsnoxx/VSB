#ifndef PATHFINDER_H
#define PATHFINDER_H

#include <vector>
#include <string>
#include "CaveSystem.h"

/**
 * @class PathFinder
 * @brief Class for finding all paths in a cave system.
 *
 * This class implements depth-first search (DFS) to find all possible paths
 * from the start cave to the end cave in a given cave system.
 */
class PathFinder {
private:
    /**
     * @brief Stores all found paths.
     */
    std::vector<std::vector<std::string>> allPaths;

    /**
     * @brief Recursive depth-first search function.
     *
     * Explores all paths from the current cave to the end.
     *
     * @param currentCave The current cave being explored.
     * @param path The current path being constructed.
     * @param caves The map of caves and their connections.
     */
    void dfs(const std::string& currentCave, std::vector<std::string>& path, const std::map<std::string, Cave>& caves);

public:
    /**
     * @brief Finds all paths in the given cave system.
     *
     * This function starts from the "start" cave and finds all paths leading to the "end" cave.
     *
     * @param caveSystem The cave system in which to search for paths.
     * @return A vector of paths, where each path is a vector of strings representing cave names.
     */
    std::vector<std::vector<std::string>> findAllPaths(const CaveSystem& caveSystem);

    /**
     * @brief Prints all paths to the standard output.
     *
     * Outputs all found paths to the console, with each path on a new line.
     */
    void printAllPaths() const;
};

#endif // PATHFINDER_H
