#include "Rotation.h"

Rotation::Rotation(float angle, const glm::vec3& axis)
    : angleFunc([angle]() { return angle; }), axis(axis) {
}

Rotation::Rotation(std::function<float()> angleFunction, const glm::vec3& axis)
    : angleFunc(angleFunction), axis(axis) {
}

glm::mat4 Rotation::getMatrix() const {
    return glm::rotate(glm::mat4(1.0f), glm::radians(angleFunc()), axis);
}