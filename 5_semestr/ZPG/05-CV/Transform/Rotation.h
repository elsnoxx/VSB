#pragma once
#include "AbstractTransform.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <functional>

class Rotation : public AbstractTransform {
public:
    // Konstruktor pro statickou rotaci
    Rotation(float angle, const glm::vec3& axis);

    // Konstruktor pro dynamickou rotaci (lambda/funkce)
    Rotation(std::function<float()> angleFunction, const glm::vec3& axis);

    // Vrací transformaèní matici
    glm::mat4 getMatrix() const override;

private:
    std::function<float()> angleFunc; // úhel v stupních
    glm::vec3 axis;                   // osa rotace
};