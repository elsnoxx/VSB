#include "HeadLight.h"
#include <iostream>

HeadLight::HeadLight(Camera* camera) : SpotLight(camera->getPosition(), camera->getTarget(), glm::vec3(1.0f, 1.0f, 1.0f), 12.5f, 17.5f), camera(camera)
{
	camera->attach(this);
	isOn = false;
}

void HeadLight::update(ObservableSubjects subject)
{
	if (subject == ObservableSubjects::SCamera)
	{
		this->position = camera->getPosition();
		this->direction = camera->getTarget();
	}

	notify(ObservableSubjects::SLight);
}