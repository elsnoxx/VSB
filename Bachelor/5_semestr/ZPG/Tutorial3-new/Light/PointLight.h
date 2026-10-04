#pragma once

#include "Light.h"
#include "LightType.h"
#include <glm/vec3.hpp>

// Point light with attenuation parameters.
// position: world-space position of the point light.
// constant, linear, quadratic: standard attenuation factors used in lighting equation.
class PointLight : public Light
{
public:
    glm::vec3 position;
    float constant;
    float linear;
    float quadratic;

    // Constructor:
    // - pos: world-space position
    // - col: light color
    // - c/l/q: attenuation factors (default values are typical for small scenes)
    PointLight(glm::vec3 pos, const glm::vec3& col, float c = 1.0f, float l = 0.09f, float q = 0.032f);
};