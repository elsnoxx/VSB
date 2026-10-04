#include "Transform.h"

// Manage a list of child transform components (scale/rotation/translation/...) that
// together form a composite local transform for a scene node.
void Transform::addTransform(std::shared_ptr<AbstractTransform> t) {
    transforms.push_back(t);
}

// Compute the local transform matrix by composing all child transforms in order.
// The multiplication order `result = result * M` means the first transform added is
// applied first to points (i.e. transforms[0] then transforms[1], ...).
// Note: when combining with parent transforms, parent * local gives the world matrix.
glm::mat4 Transform::getMatrix() const {
    glm::mat4 result(1.0f);
    for (auto& t : transforms) {
        // t->getMatrix() returns a 4x4 matrix for that primitive transform (scale/rot/translate)
        // Multiplying on the right composes transformations in the same order as added.
        result = result * t->getMatrix();
    }
    return result;
}
