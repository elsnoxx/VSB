#pragma once
#include "Light.h"
#include "LightType.h"

class DirectionalLight : public Light {
public:
    glm::vec3 direction;

    DirectionalLight(const glm::vec3 dir, const glm::vec3 col);
};

