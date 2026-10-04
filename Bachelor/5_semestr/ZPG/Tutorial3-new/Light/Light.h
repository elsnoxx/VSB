#pragma once
#include "../Observer/Subject.h"
#include "LightType.h"	

// Include GLEW
#include <GL/glew.h>
#include <glm/vec3.hpp>

// Base Light class
// - Inherits from Subject so lights can notify observers when they change.
// - Stores common light properties (type, color, intensity, on/off).
class Light : public Subject {
public:
    LightType type;        // Type of the light (directional, point, spot)
    glm::vec3 color;       // RGB color of the light
    float intensity;       // Overall intensity multiplier
    bool isOn = true;      // Enabled/disabled flag

    // Constructor
    // - type: light kind (Directional/Point/Spot)
    // - color: base color
    // - intensity: multiplier (default 1.0)
    Light(LightType type, const glm::vec3& color, float intensity = 1.0f);

    // Virtual destructor to allow safe inheritance
    virtual ~Light();

    // Return the light type
    LightType getType();
};