#include "PointLight.h"

// Construct a point light with attenuation parameters.
// Parameters:
//  - pos: world-space position
//  - col: color
//  - c/l/q: attenuation coefficients (constant, linear, quadratic)
PointLight::PointLight(glm::vec3 pos, const glm::vec3& col, float c, float l, float q)
	: Light(LightType::POINT, col), position(pos), constant(c), linear(l), quadratic(q) {
}