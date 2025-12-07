#include "ScreenManager.h"


void ScreenManager::setScenes(const std::vector<Scene*>& s) {
    scenes = s;
    currentIndex = 0;
}

void ScreenManager::switchTo(int index) {
    if (index < 0 || index >= scenes.size()) {
        printf("Invalid scene index %d\n", index);
        return;
    }

    printf("Switching to scene %d\n", index);
    currentIndex = index;
}

void ScreenManager::toggleFOV() {
    if (currentFovIndex < 3)
    {
        currentFovIndex++;
    }
    else
    {
        currentFovIndex = 0;
    }

    auto curretnScean = getCurrentScene();
    curretnScean.UpdateFOV(fovDegrees[currentFovIndex]);
}

Scene* ScreenManager::getCurrentScene() const {
    if (scenes.empty()) return nullptr;
    return scenes[currentIndex];
}

void ScreenManager::update(float dt, InputManager& input) {
    if (scenes.empty()) return;
    scenes[currentIndex]->update(dt, input);
}

void ScreenManager::draw() {
    if (scenes.empty()) return;
    scenes[currentIndex]->draw();
}

void ScreenManager::init() {
    scenes = SceneFactory::createAllScenes();
    currentIndex = 0;
}


