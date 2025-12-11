#include "ScreenManager.h"

// ScreenManager: manages multiple Scene instances and forwards high-level
// operations like update/draw and input binding to the active scene.
// Responsibilities:
// - Hold a list of available scenes and the currently active index
// - Switch between scenes, resetting scene state and re-binding input/camera
// - Forward update/draw calls from the application to the active scene
// - Change camera properties (e.g., FOV) on the active scene's camera


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

    // Obtain the newly active scene and reset its runtime state so it
    // behaves like freshly created (camera reset, control points cleared, etc.).
    Scene* scene = getCurrentScene();
    if (!scene) return;

    // Reset transient scene state (camera pose, selections, control points)
    scene->reset();

    // If an InputManager has been bound to this ScreenManager, re-bind it to
    // the new scene's camera so input continues to control the correct camera.
    if (input) {
        Camera* cam = scene->getCamera();
        if (cam) {
            input->bindCamera(cam);
            // clear internal input bookkeeping to avoid sudden mouse jumps
            input->resetState();
            input->bindScene(scene);
            // immediately push camera/light uniforms to shaders used by the scene
            scene->bindCameraAndLightToUsedShaders();
            printf("[ScreenManager] switchTo(): input bound to camera %p for scene %d\n", (void*)cam, index);
        }
    }

    // Ensure shaders are aware of the new scene's camera and lights
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


