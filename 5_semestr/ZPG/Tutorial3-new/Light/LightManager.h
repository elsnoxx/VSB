#pragma once

#include "Light.h"
#include <vector>
#define MAX_LIGHTS 16

// Simple LightManager storing raw Light* pointers.
// Note: ownership model is raw pointers here; be careful with lifetime management.
// Consider switching to smart pointers if you want automatic cleanup.
class LightManager {
private:
    std::vector<Light*> lights;
    int maxLights = MAX_LIGHTS;
public:
    // Add a light - returns pointer on success, nullptr if max capacity reached.
    Light* addLight(Light* light);

    // Get light by index. Returns nullptr for invalid index.
    Light* getLight(int id);

    // Return number of stored lights.
    int getLightsAmount();
};