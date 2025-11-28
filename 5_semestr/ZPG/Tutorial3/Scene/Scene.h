#pragma once
#include <vector>
#include "../DrawableObject.h"
#include "../Scene/Camera.h"


class Scene {
public:
    Scene(ShaderProgram* shader);

    Camera* getCamera() { return camera; }


private:
    Camera* camera;
    ShaderProgram* shaderProgram;
};

