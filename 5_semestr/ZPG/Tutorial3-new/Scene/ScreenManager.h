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
    void toggleFOV();
    Scene* getCurrentScene() const;

    void update(float dt, InputManager& input);

    void draw();

private:
    std::vector<Scene*> scenes;
    int currentIndex = 0;

    // FOV
    int currentFovIndex = 0;
    const float fovDegrees[3] = { 45.0f, 90.0f, 130.0f };
};
