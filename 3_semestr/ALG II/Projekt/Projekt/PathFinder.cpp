#include "PathFinder.h"
#include <iostream>
#include <algorithm>
#include <cctype>

/**
 * @brief Implements depth-first search to find all paths from "start" to "end".
 */
void PathFinder::dfs(const std::string& currentCave, std::vector<std::string>& path, const std::map<std::string, Cave>& caves) {
    path.push_back(currentCave);

    if (currentCave == "end") {
        allPaths.push_back(path);
    }
    else {
        for (const std::string& neighbor : caves.at(currentCave).getPassages()) {
            if (islower(neighbor[0]) && std::find(path.begin(), path.end(), neighbor) != path.end()) {
                continue;
            }
            dfs(neighbor, path, caves);
        }
    }

    path.pop_back();
}

std::vector<std::vector<std::string>> PathFinder::findAllPaths(const CaveSystem& caveSystem) {
    allPaths.clear();
    std::vector<std::string> path;
    dfs("start", path, caveSystem.getCaves());
    return allPaths;
}

void PathFinder::printAllPaths() const {
    for (const auto& path : allPaths) {
        for (const auto& cave : path) {
            std::cout << cave << " ";
        }
        std::cout << std::endl;
    }
}
