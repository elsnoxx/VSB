#include "InputManager.h"
#include <iostream>

InputManager& InputManager::instance() {
    static InputManager inst;
    return inst;
}

void InputManager::bindCamera(Camera* cam) {
    m_boundCamera = cam;
    // pokud máte jinou interní proměnnou/kód, synchronizujte ji zde
}

void InputManager::resetState() {
    // vyčistit interní stavy (příklady, upravte podle skutečných členů)
    // pressedKeys.clear();
    mouseDelta = glm::vec2(0.0f);
    lastMousePos = glm::vec2(0.0f); // Reset lastMousePos to initial state
}

void InputManager::onKey(int key, int action) {
    if (action == GLFW_PRESS) {
        keyStates[key] = true;
        keyPressedEvents[key] = true;
    }
    else if (action == GLFW_RELEASE) {
        keyStates[key] = false;
    }
}

bool InputManager::isKeyDown(int key) const {
    auto it = keyStates.find(key);
    return it != keyStates.end() && it->second;
}

bool InputManager::isKeyPressed(int key) {
    auto it = keyPressedEvents.find(key);
    if (it != keyPressedEvents.end() && it->second) {
        it->second = false; // vyzvednuto — zabráníme opětovnému vrácení true, dokud nebude nový press
        return true;
    }
    return false;
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
