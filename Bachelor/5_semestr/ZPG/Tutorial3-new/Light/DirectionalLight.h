#pragma once
#include "Light.h"
#include "LightType.h"
#include <glm/vec3.hpp>

// DirectionalLight represents a light with a direction (like the sun).
// It doesn't have a position; all lighting calculations use the direction.
class DirectionalLight : public Light {
public:
    glm::vec3 direction; // Direction vector (world-space), should be normalized.

    // Constructor:
    // - dir: light direction (points from the light toward the lit surface)
    // - col: color of the light
    DirectionalLight(const glm::vec3 dir, const glm::vec3 col);
};