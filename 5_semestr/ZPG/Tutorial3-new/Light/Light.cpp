#include "Light.h"

// Initialize base class data
Light::Light(LightType t, const glm::vec3& c, float i)
    : type(t), color(c), intensity(i)
{
}

// Return stored light type
LightType Light::getType() {
    return type;
}

// Default virtual destructor (defined out-of-line)
Light::~Light() = default;