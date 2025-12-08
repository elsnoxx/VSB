#pragma once

#include "Light.h"
#include "LightType.h"

class PointLight : public Light
{
public:
	glm::vec3 position;
	float constant;
	float linear;
	float quadratic;

	PointLight(glm::vec3 pos, const glm::vec3& col, float c = 1.0f, float l = 0.09f, float q = 0.032f);

};

