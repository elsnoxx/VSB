#include "Cave.h"

Cave::Cave(const std::string& name) : name(name) {}

void Cave::addPassage(const std::string& neighbor) {
    passages.push_back(neighbor);
}

const std::string& Cave::getName() const {
    return name;
}

const std::vector<std::string>& Cave::getPassages() const {
    return passages;
}
