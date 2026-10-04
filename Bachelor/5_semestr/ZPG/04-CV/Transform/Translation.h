#pragma once
#include "AbstractTransform.h"
#include <glm/gtc/matrix_transform.hpp>

class Translation : public AbstractTransform {
public:
    Translation(const glm::vec3& offset);
    glm::mat4 getMatrix() const override;

private:
    glm::vec3 offset;
};