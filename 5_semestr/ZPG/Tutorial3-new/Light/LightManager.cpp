#include "LightManager.h"

int LightManager::getLightsAmount() {
    return lights.size();
};


Light* LightManager::addLight(Light* light) {
    if (lights.size() > maxLights)
        return nullptr;

    lights.push_back(light);
    return light;
}


Light* LightManager::getLight(int id) {
    if (id < 0 || id > lights.size())
        return nullptr;

    return lights[id];
}