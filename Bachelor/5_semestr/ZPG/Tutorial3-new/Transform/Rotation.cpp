#include "Rotation.h"

// Rotation transform wrapper
// - `angleFunc` is a callable that returns the rotation angle in degrees (commonly used in the codebase).
// - `axis` is the axis of rotation (preferably normalized). glm::rotate expects a normalized axis
//   for the most predictable results; passing a non-normalized axis will effectively scale the rotation.
//
// There are two constructors:
// 1) constant angle: captures a fixed angle value and wraps it into a function returning that value.
// 2) dynamic angle: accepts a std::function<float()> so the rotation can change every frame (animation).
Rotation::Rotation(float angle, const glm::vec3& axis)
    : angleFunc([angle]() { return angle; }), axis(axis) {
}

Rotation::Rotation(std::function<float()> angleFunction, const glm::vec3& axis)
    : angleFunc(angleFunction), axis(axis) {
}

// Build the 4x4 rotation matrix.
// Note: glm::rotate expects the angle in radians, so we convert degrees->radians here using glm::radians().
// The resulting matrix rotates points around `axis` by `angleFunc()` degrees.
glm::mat4 Rotation::getMatrix() const {
    return glm::rotate(glm::mat4(1.0f), glm::radians(angleFunc()), axis);
}