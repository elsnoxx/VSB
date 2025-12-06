#pragma once
#include <unordered_map>
#include <glm/vec2.hpp>
#include <GLFW/glfw3.h>

#include "../Config.h"

#include "../Scene/ScreenManager.h"

class ScreenManager;


class InputManager {
public:
    void onKey(int key, int action);
    void onMouseMove(double x, double y);
    glm::vec2 getMouseDeltaAndReset(float dt);

    void OnMouseClick(double x, double y);

    bool isKeyDown(int key) const;

    glm::vec2 getMouseDelta();
    void endFrame();

    void setScreenManager(ScreenManager* sm);

private:
    std::unordered_map<int, bool> keyStates;
    
    ScreenManager* screenManager = nullptr;

    glm::vec2 lastMousePos = { 0,0 };
    glm::vec2 mouseDelta = { 0,0 };
    bool firstMouse = true;
};
