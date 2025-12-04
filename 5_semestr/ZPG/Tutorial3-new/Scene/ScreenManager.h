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

    void draw();

private:
    std::vector<Scene*> scenes;
    int currentIndex = 0;
};
