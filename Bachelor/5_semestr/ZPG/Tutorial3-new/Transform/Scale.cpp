#include "Scale.h"

// Scale transform
// - `factors` holds the scaling factors along X,Y,Z (can be non-uniform).
// - The constructor here accepts a constant scale vector. If you need an
//   animated or time-dependent scale, consider adding a constructor that
//   accepts a `std::function<glm::vec3()>` similar to Rotation/Translation.
//
// Note: `glm::scale` produces a matrix that scales in model space. When combining
// transforms, matrix multiplication order matters: `T * R * S` applies scale first,
// then rotation, then translation (when a point is multiplied on the right).
Scale::Scale(const glm::vec3& factors) : factors(factors) {
}

glm::mat4 Scale::getMatrix() const {
    // Build scale matrix from factors; passing a non-uniform vector scales each axis separately.
    return glm::scale(glm::mat4(1.0f), factors);
}