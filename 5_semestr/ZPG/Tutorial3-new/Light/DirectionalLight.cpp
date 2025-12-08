#include "DirectionalLight.h"

DirectionalLight::DirectionalLight(const glm::vec3 dir, const glm::vec3 col)
	: Light(LightType::DIRECTIONAL, col), direction(dir) {
}