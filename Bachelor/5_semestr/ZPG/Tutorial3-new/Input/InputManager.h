#pragma once
#include <unordered_map>
#include "../Camera/Camera.h"
#include <glm/vec2.hpp>
#include <GLFW/glfw3.h>

#include "../Config.h"
#include "../ModelObject/ModelType.h"
#include "../Shader/ShaderType.h"

// InputManager is a simple singleton that collects keyboard and mouse events,
// exposes polling helpers and forwards mouse orientation to a bound Camera.
// Typical usage:
// - call onKey/onMouseMove from the GLFW callbacks,
// - call getMouseDeltaAndReset(dt) each frame to obtain per-second mouse offset,
// - use isKeyDown/isKeyPressed to react to keys in the game loop.
class InputManager {
public:
    // Get the singleton instance.
    static InputManager& instance();

    // Called from the platform (GLFW) key callback.
    // - key: GLFW key code
    // - action: GLFW_PRESS / GLFW_RELEASE / GLFW_REPEAT
    // This updates internal key state and one-shot "pressed" events.
    void onKey(int key, int action);

    // Called from the platform mouse-move callback.
    // - x,y are absolute mouse positions in window coordinates.
    // The manager computes delta between consecutive calls and accumulates it.
    void onMouseMove(double x, double y);

    // Returns the mouse movement delta (normalized by dt) and resets the internal accumulator.
    // - dt: frame time in seconds; the function typically returns delta/dt so caller gets
    //       mouse movement per second (useful for multiplying by sensitivity and time).
    // After this call the accumulated mouse delta is cleared.
    glm::vec2 getMouseDeltaAndReset(float dt);

    // Key queries:
    // - isKeyDown(key): returns true while the key is held down.
    // - isKeyPressed(key): returns true only once per press (edge-triggered).
    bool isKeyDown(int key) const;
    bool isKeyPressed(int key);

    // Return current (not-reset) mouse delta accumulated since last reset.
    glm::vec2 getMouseDelta();

    // Called at end of frame to clear one-frame events and perform housekeeping.
    // Should be called once per frame by the main loop.
    void endFrame();

    // Bind a Camera to the input manager: if set, the InputManager can forward
    // mouse deltas to the camera (or caller can obtain deltas and call camera API).
    // Note: InputManager does not take ownership of the Camera pointer.
    void bindCamera(Camera* cam);

    // Bind the active Scene so input manager can operate on scene-level actions
    // (object placement, picking, etc). Non-owning pointer.
    void bindScene(class Scene* scene);

    // Placement API: start placing objects of given model type using given shader.
    // While placement mode is active, mouse click will plant objects in the bound scene.
    void startPlacement(ModelType model, ShaderType shader);
    void stopPlacement();

    // Handle mouse button click from application (x,y in window coordinates).
    // This forwards to placement/picking logic when appropriate.
    void onMouseButton(double x, double y, int button);

    // Query placement mode
    bool isPlacing() const { return placing; }

    // Reset internal state (clear key states, mouse deltas, pressed events).
    // Useful when switching scenes or when focus is lost.
    void resetState();

private:
    // Map of current key states (true = key currently down)
    std::unordered_map<int, bool> keyStates;

    // Map of edge-triggered "pressed" events. When a key is pressed this is set true
    // and remains true until consumed by isKeyPressed() or cleared in endFrame().
    std::unordered_map<int, bool> keyPressedEvents;

    // Pointer to camera controlled by input (non-owning).
    Camera* m_boundCamera = nullptr;

    // Pointer to bound scene for placement/picking actions.
    class Scene* m_boundScene = nullptr;

    // Placement state
    bool placing = false;
    ModelType placementModel = ModelType::Tree;
    ShaderType placementShader = ShaderType::Phong;

    // Mouse tracking:
    // - lastMousePos: last absolute mouse position seen (window coords).
    // - mouseDelta: accumulated delta since last reset (in pixels).
    // - firstMouse: helper to ignore large initial jump on first callback.
    glm::vec2 lastMousePos = { 0,0 };
    glm::vec2 mouseDelta = { 0,0 };
    bool firstMouse = true;
};