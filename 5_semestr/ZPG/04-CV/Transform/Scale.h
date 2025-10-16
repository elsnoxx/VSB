#pragma once
#include "AbstractTransform.h"
#include <glm/gtc/matrix_transform.hpp>

class Scale : public AbstractTransform {
public:
    Scale(const glm::vec3& factors);
    virtual glm::mat4 getMatrix() const override;  // Add override keyword

private:
    glm::vec3 factors;
};