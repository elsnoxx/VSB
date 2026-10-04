#include "Transform.h"

void Transform::addTransform(std::shared_ptr<AbstractTransform> t) {
    transforms.push_back(t);
}

glm::mat4 Transform::getMatrix() const {
    glm::mat4 result(1.0f);
    for (auto& t : transforms) {
        result = result * t->getMatrix(); // poøadí je dùležité!
    }
    return result;
}
