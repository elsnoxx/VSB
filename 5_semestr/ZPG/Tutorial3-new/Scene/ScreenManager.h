#pragma once
#include <vector>
#include <cstdio>
#include "Scene.h"
#include "SceneFactory.h"
#include "../Shader/ShaderProgram.h"
#include "../Input/InputManager.h"

class ScreenManager {
public:
    ScreenManager() = default;

    void init();
    void setScenes(const std::vector<Scene*>& scenes);

    void switchTo(int index);

    Scene* getCurrentScene() const;

    void update(float dt, InputManager& input);

    void changeFOV(float radians);

    void draw();

    // Bind the application's InputManager so ScreenManager can reset/bind it on switches.
    void bindInput(InputManager* inputMgr) { input = inputMgr; }

private:
    std::vector<Scene*> scenes;
    int currentIndex = 0;

    // FOV
    int currentFovIndex = 0;

    // pointer to external InputManager (non-owning)
    InputManager* input = nullptr;
};