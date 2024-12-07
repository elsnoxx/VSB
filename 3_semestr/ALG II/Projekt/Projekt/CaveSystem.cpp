#include "CaveSystem.h"
#include <fstream>
#include <stdexcept>

void CaveSystem::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        size_t dashPos = line.find("-");
        std::string cave1 = line.substr(0, dashPos);
        std::string cave2 = line.substr(dashPos + 1);

        if (caves.find(cave1) == caves.end()) {
            caves[cave1] = Cave(cave1);
        }
        if (caves.find(cave2) == caves.end()) {
            caves[cave2] = Cave(cave2);
        }

        caves[cave1].addPassage(cave2);
        caves[cave2].addPassage(cave1);
    }
}

const std::map<std::string, Cave>& CaveSystem::getCaves() const {
    return caves;
}
