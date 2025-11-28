#pragma once
#include "Model.h"
#include "./Shader/ShaderProgram.h"
#include "./Transform/Transform.h"

class DrawableObject {
public:
    DrawableObject(Model* m, ShaderProgram* s)
        : model(m), shader(s) {
    }

    void setTransform(const Transform& t) { transform = t; }

    void draw();

private:
    Model* model;
    ShaderProgram* shader;
    Transform transform;
};
