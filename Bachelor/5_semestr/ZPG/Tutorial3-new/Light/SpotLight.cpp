#include "SpotLight.h"
#include <glm/gtc/constants.hpp>

// SpotLight constructor computes cosines of cutoff angles for shader use.
SpotLight::SpotLight(const glm::vec3& pos, const glm::vec3& dir, const glm::vec3& col,
    float cutDeg, float outerCutDeg)
    : Light(SPOT, col),
      position(pos), direction(dir)
{
    // precompute cosines of cutoff angles to compare with dot(direction, lightDir) in shader
    cutOff = glm::cos(glm::radians(cutDeg));
    outerCutOff = glm::cos(glm::radians(outerCutDeg));
}