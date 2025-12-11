#include "LightManager.h"

// Return number of lights in the manager
int LightManager::getLightsAmount() {
    return lights.size();
};

// Add a light if we have capacity. Returns the light pointer when added.
Light* LightManager::addLight(Light* light) {
    if (lights.size() > maxLights)
        return nullptr;

    lights.push_back(light);
    return light;
}

// Return pointer to light at id, or nullptr if out-of-range.
// Note: callers must ensure they do not delete lights unexpectedly since manager
// stores raw pointers.
Light* LightManager::getLight(int id) {
    if (id < 0 || id > lights.size())
        return nullptr;

    return lights[id];
}