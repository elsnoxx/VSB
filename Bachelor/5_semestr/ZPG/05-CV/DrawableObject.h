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

    void draw() {
        shader->use();
        shader->setUniformMat4("modelMatrix", transform.getMatrix());
        model->draw(); // volá vlastní draw metodu modelu, která ví, jak použít VAO/VBO
    }

private:
    Model* model;
    ShaderProgram* shader;
    Transform transform;
};
