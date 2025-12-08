#pragma once
#include "Light.h"
#include "LightType.h"

#include <glm/gtc/matrix_transform.hpp>

class SpotLight : public Light
{
public:
    glm::vec3 position;
    glm::vec3 direction;
    float cutOff;
    float outerCutOff;

    SpotLight(const glm::vec3& pos, const glm::vec3& dir, const glm::vec3& col, float cutDeg, float outerCutDeg);

};

