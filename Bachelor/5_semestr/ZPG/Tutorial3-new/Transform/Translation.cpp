#include "Translation.h"

// Translation transform
// - `offset` is the translation vector applied in model/local space.
// - The constructor takes a constant offset. For animated translation, consider
//   adding a constructor that accepts `std::function<glm::vec3()>` like other transforms.
//
// Note: `glm::translate` produces a matrix that translates points by `offset`.
// When composing transforms, the multiplication order determines whether translation
// happens before or after rotation/scale (e.g., T * R * S applies scale first,
// then rotation, then translation when multiplying a column-vector on the right).
Translation::Translation(const glm::vec3& offset) : offset(offset) {
}

glm::mat4 Translation::getMatrix() const {
    return glm::translate(glm::mat4(1.0f), offset);
}