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

    // get new scene
    Scene* scene = getCurrentScene();
    if (!scene) return;

    // Reset scene internals so it behaves like freshly created
    scene->reset();

    // If an external InputManager was bound to this ScreenManager, attach it.
    if (input) {
        Camera* cam = scene->getCamera();
        if (cam) {
            input->bindCamera(cam);
            input->resetState(); // ensure firstMouse and lastMouse don't cause jumps
            input->bindScene(scene);
            scene->bindCameraAndLightToUsedShaders();
            printf("[ScreenManager] switchTo(): input bound to camera %p for scene %d\n", (void*)cam, index);
        }
    }

    // Update shaders with camera/light uniforms for the newly active scene
    scene->bindCameraAndLightToUsedShaders();
}


void ScreenManager::changeFOV(float radians) {
    Scene * cur = getCurrentScene();
    if (!cur) return;
    Camera * cam = cur->getCamera();
    if (!cam) return;
    cam->setFOV(radians);
        // save index or log if needed
    printf("[ScreenManager] Changed FOV of scene %d to %f radians\n", currentIndex, radians);
    
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

    // bind initial scene camera
    Scene* scene = getCurrentScene();
    if (scene) {
        Camera* cam = scene->getCamera();
        if (cam && input) {
            input->bindCamera(cam);
            input->bindScene(scene);
            input->resetState();
        }
    }
}


