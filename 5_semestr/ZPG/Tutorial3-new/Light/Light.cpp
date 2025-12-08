#include "Light.h"

Light::Light(LightType t, const glm::vec3& c, float i)
	: type(t), color(c), intensity(i)
{
}

LightType Light::getType() {
	return type;
}

Light::~Light() = default;