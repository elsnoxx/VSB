#include "InputManager.h"
#include <iostream>
#include "../Scene/Scene.h"
#include "../ModelObject/ModelType.h"
#include "../Shader/ShaderType.h"

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

// Bind a Scene pointer (non-owning). Used for placement/picking.
void InputManager::bindScene(Scene* scene) {
    m_boundScene = scene;
}

// Start placing objects of given model and shader type.
void InputManager::startPlacement(ModelType model, ShaderType shader) {
    placing = true;
    placementModel = model;
    placementShader = shader;
}

void InputManager::stopPlacement() {
    placing = false;
}

// Handle mouse button - called from GLFW mouse button callback.
void InputManager::onMouseButton(double x, double y, int button) {
    // need scene + camera bound to do picking/placement
    if (!m_boundScene || !m_boundCamera) return;

    glm::vec3 worldPos;
    int picked = m_boundScene->pickAtCursor(x, y, &worldPos);

    if (picked >= 0) {
        printf("[InputManager] picked object index %d at world [%f,%f,%f]\n", picked, worldPos.x, worldPos.y, worldPos.z);

        if (placing) {
            // placement-specific behavior: plant selected model / remove / add control point
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                m_boundScene->plantObjectAtWorldPos(worldPos, placementModel, placementShader);
            }
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                bool ok = m_boundScene->removeObjectAt(picked);
                printf("[InputManager] removeObjectAt(%d) -> %s\n", picked, ok ? "ok" : "failed");
            }
            else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                m_boundScene->addControlPoint(worldPos);
                printf("[InputManager] added control point [%f,%f,%f] (total %zu)\n", worldPos.x, worldPos.y, worldPos.z, m_boundScene->getControlPoints().size());
                const auto& pts = m_boundScene->getControlPoints();
                if (pts.size() >= 4 && (pts.size() % 4) == 0) {
                    m_boundScene->buildBezierFromControlPoints(6.0f, true);
                    printf("[InputManager] built Bezier segment(s) from control points\n");
                }
            }
        }
        else {
            // non-placement (default app behavior) — replicate former Application::handleMouseClick actions
            if (button == GLFW_MOUSE_BUTTON_LEFT) {
                m_boundScene->plantObjectAtWorldPos(worldPos, ModelType::Tree, ShaderType::Phong);
            }
            else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
                bool ok = m_boundScene->removeObjectAt(picked);
                printf("[InputManager] removeObjectAt(%d) -> %s\n", picked, ok ? "ok" : "failed");
            }
            else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
                m_boundScene->addControlPoint(worldPos);
                printf("[InputManager] added control point [%f,%f,%f] (total %zu)\n", worldPos.x, worldPos.y, worldPos.z, m_boundScene->getControlPoints().size());
                const auto& pts = m_boundScene->getControlPoints();
                if (pts.size() >= 4 && (pts.size() % 4) == 0) {
                    m_boundScene->buildBezierFromControlPoints(6.0f, true);
                    printf("[InputManager] built Bezier segment(s) from control points\n");
                }
            }
        }
    }
    else {
        // Clicked empty space -> intersect with y=0 plane, behave similarly for placing/non-placing
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            int fbw, fbh; glfwGetFramebufferSize(glfwGetCurrentContext(), &fbw, &fbh);
            glm::vec3 nearP((float)x, (float)(fbh - y), 0.0f);
            glm::vec3 farP((float)x, (float)(fbh - y), 1.0f);
            glm::mat4 view = m_boundScene->getCamera()->getViewMatrix();
            glm::mat4 proj = m_boundScene->getCamera()->getProjectionMatrix();
            glm::vec4 vp(0, 0, (float)fbw, (float)fbh);
            glm::vec3 n = glm::unProject(nearP, view, proj, vp);
            glm::vec3 f = glm::unProject(farP, view, proj, vp);
            glm::vec3 dir = glm::normalize(f - n);
            if (fabs(dir.y) > 1e-6f) {
                float t = -n.y / dir.y;
                if (t > 0.0f) {
                    glm::vec3 planePos = n + dir * t;
                    if (placing) {
                        m_boundScene->plantObjectAtWorldPos(planePos, placementModel, placementShader);
                    }
                    else {
                        m_boundScene->plantObjectAtWorldPos(planePos, ModelType::Tree, ShaderType::Phong);
                    }
                }
            }
        }
        else if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            int fbw, fbh; glfwGetFramebufferSize(glfwGetCurrentContext(), &fbw, &fbh);
            glm::vec3 nearP((float)x, (float)(fbh - y), 0.0f);
            glm::vec3 farP((float)x, (float)(fbh - y), 1.0f);
            glm::mat4 view = m_boundScene->getCamera()->getViewMatrix();
            glm::mat4 proj = m_boundScene->getCamera()->getProjectionMatrix();
            glm::vec4 vp(0, 0, (float)fbw, (float)fbh);
            glm::vec3 n = glm::unProject(nearP, view, proj, vp);
            glm::vec3 f = glm::unProject(farP, view, proj, vp);
            glm::vec3 dir = glm::normalize(f - n);
            if (fabs(dir.y) > 1e-6f) {
                float t = -n.y / dir.y;
                if (t > 0.0f) {
                    glm::vec3 planePos = n + dir * t;
                    m_boundScene->addControlPoint(planePos);
                    printf("[InputManager] added control point on plane [%f,%f,%f] (total %zu)\n", planePos.x, planePos.y, planePos.z, m_boundScene->getControlPoints().size());
                    const auto& pts = m_boundScene->getControlPoints();
                    if (pts.size() >= 4 && (pts.size() % 4) == 0) {
                        m_boundScene->buildBezierFromControlPoints(0.25f, true);
                        printf("[InputManager] built Bezier segment(s) from control points\n");
                    }
                }
            }
        }
    }
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
        printf("[InputManager] onMouseMove() firstMouse init pos=(%f,%f)\n", x, y);
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

    if (delta.x != 0.0f || delta.y != 0.0f) {
        printf("[InputManager] onMouseMove() delta=(%f,%f) last=(%f,%f)\n", delta.x, delta.y, lastMousePos.x, lastMousePos.y);
    }
    // Update last position
    lastMousePos = current;
}

// Return the raw accumulated mouse delta (in pixels) since last reset.
// Does not modify internal state.
glm::vec2 InputManager::getMouseDelta() {
    return mouseDelta;
}

// Return scaled mouse delta and reset accumulator.
// - dt: frame delta time in seconds. We apply Config::MouseSensitivity here but do NOT
//   scale by dt; callers (Camera::updateOrientation) should apply dt to rotate consistently.
glm::vec2 InputManager::getMouseDeltaAndReset(float /*dt*/) {
    glm::vec2 delta = mouseDelta * Config::MouseSensitivity;
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