#pragma once
#include "../Observer/Subject.h"
#include "LightType.h"	

//Include GLEW
#include <GL/glew.h>
#include <glm/vec3.hpp>

class Light : public Subject {
public:
    LightType type;
    glm::vec3 color;
    float intensity;
    bool isOn = true;

    Light(LightType type, const glm::vec3& color, float intensity = 1.0f);
    virtual ~Light();

    LightType getType();
};