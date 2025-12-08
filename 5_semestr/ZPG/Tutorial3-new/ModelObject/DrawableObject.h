#pragma once
#include "Model.h"
#include "../Shader/ShaderProgram.h"
#include "../Transform/Transform.h"
#include "../Shader/ShaderType.h"
#include "../Shader/ShaderFactory.h"

class DrawableObject {
public:
    DrawableObject(Model* m, ShaderType shaderType);

    void setTransform(const Transform& t) { transform = t; }
    Transform& getTransform() { return transform; }
    ShaderProgram* getShader() const { return shaderProgram; }

    void draw();

    void setTexture(GLuint tex) { textureID = tex; }
    bool hasTexture() const { return textureID != 0; }

protected:
	ShaderProgram* shaderProgram = nullptr;
    Model* model = nullptr;
    Transform transform;
    GLuint textureID = 0;

};

