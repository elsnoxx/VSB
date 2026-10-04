#pragma once
#include <glm/glm.hpp>

class AbstractTransform {
public:
    virtual ~AbstractTransform() {}
    virtual glm::mat4 getMatrix() const = 0;  // Add const here
};