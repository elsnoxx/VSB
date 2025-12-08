#include "PointLight.h"

PointLight::PointLight(glm::vec3 pos, const glm::vec3& col, float c, float l, float q)
	: Light(LightType::POINT, col), position(pos), constant(c), linear(l), quadratic(q) {
}