#include "InputManager.h"


void InputManager::setScreenManager(ScreenManager* sm) {
    screenManager = sm;
}

void InputManager::OnMouseClick(double x, double y) {
    if (!screenManager) return;
    Scene* cur = screenManager->getCurrentScene();
    if (!cur) return;

    Camera* cam = cur->getCamera();
    if (cam) {
        glm::vec3 spawn = cam->getPosition() + cam->getPosition() * 5.0f + glm::vec3(0.0f, -1.0f, 0.0f);
        cur->spawnTree(spawn);
        return;
    }

    // fallback: použít původní chování (pokud žádná kamera)
    cur->spawnTree(glm::vec3((float)x, 0.0f, (float)y));
}

void InputManager::onKey(int key, int action) {
    if (action == GLFW_PRESS)
        keyStates[key] = true;
    else if (action == GLFW_RELEASE)
        keyStates[key] = false;
}

bool InputManager::isKeyDown(int key) const {
    auto it = keyStates.find(key);
    return it != keyStates.end() && it->second;
}


void InputManager::onMouseMove(double x, double y) {
    // if (firstMouse) {
    //     lastMousePos = { x, y };
    //     firstMouse = false;
    // }

    mouseDelta = { (float)(x - lastMousePos.x), (float)(lastMousePos.y - y) };
    lastMousePos = { x, y };
}

glm::vec2 InputManager::getMouseDelta() {
    return mouseDelta;
}

glm::vec2 InputManager::getMouseDeltaAndReset(float dt) {
    glm::vec2 delta = mouseDelta * Config::MouseSensitivity;
    delta *= dt;
    mouseDelta = { 0, 0 };
    return delta;
}

void InputManager::endFrame() {
    mouseDelta = { 0,0 };
}
