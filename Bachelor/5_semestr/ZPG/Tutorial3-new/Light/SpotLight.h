#pragma once
#include "Light.h"
#include "LightType.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/vec3.hpp>

// SpotLight: a cone-shaped light with inner and outer cutoff angles.
// - position: world-space origin of the cone
// - direction: forward direction of the cone (should be normalized)
// - cutOff / outerCutOff: cosines of inner and outer cone angles (set in constructor)
class SpotLight : public Light
{
public:
    glm::vec3 position;
    glm::vec3 direction;
    float cutOff;      // inner cutoff cosine
    float outerCutOff; // outer cutoff cosine

    // Constructor:
    // - pos: position
    // - dir: direction
    // - col: color
    // - cutDeg: inner cone angle in degrees
    // - outerCutDeg: outer cone angle in degrees
    SpotLight(const glm::vec3& pos, const glm::vec3& dir, const glm::vec3& col, float cutDeg, float outerCutDeg);
};