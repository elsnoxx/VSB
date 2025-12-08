#pragma once

#include "Light.h"
#include <vector>
#define MAX_LIGHTS 16

class LightManager {
private:
    std::vector<Light*> lights;
    int maxLights = MAX_LIGHTS;
public:
    Light* addLight(Light* light); //addd light, if added return pointer ; other: nullptr
    Light* getLight(int id);
    int getLightsAmount();
};

