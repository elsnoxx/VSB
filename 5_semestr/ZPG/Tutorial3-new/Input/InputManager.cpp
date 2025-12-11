#include "InputManager.h"
#include <iostream>

// Singleton accessor
InputManager& InputManager::instance() {
    static InputManager inst;
    return inst;
}

// Bind a camera pointer (non-owning). InputManager can forward mouse deltas to this camera.
void InputManager::bindCamera(Camera* cam) {
    m_boundCamera = cam;
    // If additional synchronization is needed (e.g. resetting mouse state), do it here.
}

// Reset internal input state (useful when switching scenes or losing focus).
void InputManager::resetState() {
    // Clear accumulated mouse delta and reset last mouse position so next movement won't jump.
    mouseDelta = glm::vec2(0.0f);
    lastMousePos = glm::vec2(0.0f);
    firstMouse = true; // ensure first subsequent mouse event is handled as initial position
    keyStates.clear();
    keyPressedEvents.clear();
}

// Key callback entry point (should be called from GLFW key callback).
// - key: GLFW key code
// - action: GLFW_PRESS / GLFW_RELEASE / GLFW_REPEAT
void InputManager::onKey(int key, int action) {
    if (action == GLFW_PRESS) {
        keyStates[key] = true;
        keyPressedEvents[key] = true; // edge triggered press event
    }
    else if (action == GLFW_RELEASE) {
        keyStates[key] = false;
    }
    // Note: GLFW_REPEAT typically treated as "key still down" (keyStates remains true)
}

// Query whether a key is currently held down.
bool InputManager::isKeyDown(int key) const {
    auto it = keyStates.find(key);
    return it != keyStates.end() && it->second;
}

// Edge-triggered pressed query: returns true only once per physical press.
// After returning true the event is consumed and won't fire again until the next press.
bool InputManager::isKeyPressed(int key) {
    auto it = keyPressedEvents.find(key);
    if (it != keyPressedEvents.end() && it->second) {
        it->second = false; // consume event
        return true;
    }
    return false;
}

// Mouse movement callback (absolute coordinates from GLFW).
// We compute delta as difference to lastMousePos and store it in mouseDelta.
// The first mouse event is used to initialize lastMousePos to avoid large jump.
void InputManager::onMouseMove(double x, double y) {
    // Handle the first mouse callback specially to avoid huge initial delta.
    if (firstMouse) {
        lastMousePos = { (float)x, (float)y };
        firstMouse = false;
        mouseDelta = { 0.0f, 0.0f };
        return;
    }

    // Note: we return delta in (x, y) where y is typically inverted for many apps.
    // Current convention: mouseDelta.x = deltaX, mouseDelta.y = deltaY (positive up).
    glm::vec2 current = { (float)x, (float)y };
    glm::vec2 delta = current - lastMousePos;

    // Store delta in pixels (accumulated until getMouseDeltaAndReset or endFrame).
    mouseDelta += delta;

    // Update last position
    lastMousePos = current;
}

// Return the raw accumulated mouse delta (in pixels) since last reset.
// Does not modify internal state.
glm::vec2 InputManager::getMouseDelta() {
    return mouseDelta;
}

// Return scaled mouse delta and reset accumulator.
// - dt: frame delta time in seconds. We apply Config::MouseSensitivity and scale by dt here,
//   so callers receive a delta ready to multiply into rotation/speed calculations.
glm::vec2 InputManager::getMouseDeltaAndReset(float dt) {
    glm::vec2 delta = mouseDelta * Config::MouseSensitivity;
    // Scale by dt to produce per-frame, time-corrected displacement
    delta *= dt;
    // reset accumulator
    mouseDelta = { 0.0f, 0.0f };
    return delta;
}

// End-of-frame housekeeping.
// Currently resets mouseDelta (defensive) — getMouseDeltaAndReset already resets delta,
// but calling endFrame ensures all per-frame accumulators are cleared.
void InputManager::endFrame() {
    mouseDelta = { 0.0f, 0.0f };
    // Clear one-shot key pressed events if you prefer to consume them at endFrame:
    // keyPressedEvents.clear();
}