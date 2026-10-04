#include "Light.h"

// Initialize base class data
// Parameters:
//  - t: light type (Directional/Point/Spot)
//  - c: RGB color of the light
//  - i: intensity multiplier
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