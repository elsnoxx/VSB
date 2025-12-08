#include "SpotLight.h"



SpotLight::SpotLight(const glm::vec3& pos, const glm::vec3& dir, const glm::vec3& col,
    float cutDeg = 12.5f, float outerCutDeg = 17.5f)
    : Light(SPOT, col),
    position(pos), direction(dir)
{
    cutOff = glm::cos(glm::radians(cutDeg));
    outerCutOff = glm::cos(glm::radians(outerCutDeg));
}