#include "LightManager.h"

// Return number of lights currently stored in the manager.
int LightManager::getLightsAmount() {
    return static_cast<int>(lights.size());
};

// Add a light if there's capacity. Returns the same pointer on success, or
// nullptr if the manager is full.
// Note: ownership remains with the caller; this manager stores raw pointers.
Light* LightManager::addLight(Light* light) {
    if (static_cast<int>(lights.size()) > maxLights)
        return nullptr;

    lights.push_back(light);
    return light;
}

// Return pointer to light at index `id`, or nullptr if out-of-range.
// Callers must not delete the returned pointer while it is stored in the manager.
Light* LightManager::getLight(int id) {
    if (id < 0 || id >= static_cast<int>(lights.size()))
        return nullptr;

    return lights[id];
}