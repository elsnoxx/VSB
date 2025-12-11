#include "DirectionalLight.h"

// Create a directional light with given direction and color.
// Note: We don't normalize direction here; caller should provide a normalized vector
// or shader code / usage will normalize it as needed.
DirectionalLight::DirectionalLight(const glm::vec3 dir, const glm::vec3 col)
    : Light(LightType::DIRECTIONAL, col), direction(dir) {
}