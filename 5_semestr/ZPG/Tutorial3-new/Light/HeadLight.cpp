#include "HeadLight.h"
#include <iostream>

// HeadLight: a spotlight that follows the camera. It observes the camera and
// updates its world-space `position` and `direction` whenever the camera moves.
// The constructor initializes the spot parameters (color and cone angles) and
// attaches this HeadLight to the camera observer list.
HeadLight::HeadLight(Camera* camera)
    : SpotLight(camera->getPosition(), camera->getTarget(), glm::vec3(1.0f,1.0f,1.0f), 12.5f, 17.5f),
      camera(camera)
{
    // Attach this HeadLight as observer of the camera so update() is called automatically.
    camera->attach(this);

    // Start with the headlight turned off; toggle elsewhere in code if needed.
    isOn = false;
}

// Observer callback: called when observed subjects notify. We expect the camera
// to send `ObservableSubjects::SCamera` when its transform changed, so update
// the headlight origin and forward direction accordingly.
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