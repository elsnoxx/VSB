#pragma once

#include "SpotLight.h"
#include "../Observer/Observer.h"
#include "../Observer/ObservableSubjects.h"
#include "../Camera/Camera.h"

// HeadLight is a spot light attached to a Camera. It observes the Camera
// and updates its position and direction whenever the camera moves/orients.
class HeadLight : public SpotLight, public Observer
{
private:
    Camera* camera; // non-owning pointer to the camera

    // Observer callback: called when observed subjects (camera) notify.
    void update(ObservableSubjects subject) override;

public:
    // Create headlight and attach it to the camera observer list.
    // The headlight will track the camera position and forward direction.
    HeadLight(Camera* camera);
};