#include "InputManager.h"



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
    if (firstMouse) {
        lastMousePos = { x, y };
        firstMouse = false;
    }

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
