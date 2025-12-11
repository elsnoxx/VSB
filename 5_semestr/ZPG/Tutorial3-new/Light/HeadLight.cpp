#include "HeadLight.h"
#include <iostream>

// Create a HeadLight that follows the camera.
// We use SpotLight constructor with camera position and target as initial values.
HeadLight::HeadLight(Camera* camera)
    : SpotLight(camera->getPosition(), camera->getTarget(), glm::vec3(1.0f,1.0f,1.0f), 12.5f, 17.5f),
      camera(camera)
{
    // Attach this HeadLight as observer of the camera so update() is called automatically.
    camera->attach(this);

    // Start with the headlight turned off; toggle elsewhere in code if needed.
    isOn = false;
}

// Observer update: when camera notifies, update the headlight position/direction.
void HeadLight::update(ObservableSubjects subject)
{
    if (subject == ObservableSubjects::SCamera)
    {
        // Align light origin and direction with the camera.
        this->position = camera->getPosition();
        this->direction = camera->getTarget();
    }

    // Notify listeners that the light changed (if any).
    notify(ObservableSubjects::SLight);
}