#pragma once
#include <unordered_map>
#include "../Camera/Camera.h"
#include <glm/vec2.hpp>
#include <GLFW/glfw3.h>

#include "../Config.h"


//#include "../Scene/ScreenManager.h"

//class ScreenManager;


class InputManager {
public:
    static InputManager& instance();

    void onKey(int key, int action);
    void onMouseMove(double x, double y);
    glm::vec2 getMouseDeltaAndReset(float dt);

    bool isKeyDown(int key) const;
    bool isKeyPressed(int key);

    glm::vec2 getMouseDelta();
    void endFrame();

    void bindCamera(Camera* cam);
    void resetState();

private:
    std::unordered_map<int, bool> keyStates;
    std::unordered_map<int, bool> keyPressedEvents;
    
    Camera* m_boundCamera = nullptr;

    glm::vec2 lastMousePos = { 0,0 };
    glm::vec2 mouseDelta = { 0,0 };
    bool firstMouse = true;
};
