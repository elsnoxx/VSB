#pragma once
#include <GL/glew.h>
#include "Model.h"
#include "../Shader/ShaderFactory.h"
#include "TextureType.h"
#include "../Transform/Transform.h"
#include <memory>

class DrawableObject {
public:
    DrawableObject(Model* m, ShaderType shaderType, TextureType textureType = TextureType::Empty);

    void setTransform(const Transform& t) { transform = t; }
    Transform& getTransform() { return transform; }
    ShaderProgram* getShader() const { return shaderProgram; }
    
    void setTextureID(GLuint id);
    GLuint getTextureID() const;

    void setTextureType(TextureType t);
    TextureType getTextureType() const;

    void draw();

    Transform transform;

private:
    Model* model = nullptr;
    ShaderProgram* shaderProgram = nullptr;
    GLuint textureID = 0;
    TextureType textureType = TextureType::Empty;
};