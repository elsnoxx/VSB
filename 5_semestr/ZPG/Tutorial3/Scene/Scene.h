#pragma once
#include <vector>
#include "../Model/DrawableObject.h"
#include "../Camera/Camera.h"
#include "../Shader/ShaderProgram.h"

class Scene {
public:
    Scene(ShaderProgram* shader);
    void addObject(DrawableObject* obj);
    void draw();

    Camera* getCamera() { return camera; }

private:
    Camera* camera;
    ShaderProgram* shaderProgram;
    std::vector<DrawableObject*> objects;
};
