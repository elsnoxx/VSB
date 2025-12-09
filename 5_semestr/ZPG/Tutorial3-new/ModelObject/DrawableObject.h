#pragma once
#include <GL/glew.h>
#include <memory>
#include "Model.h"
#include "../Shader/ShaderFactory.h"
#include "TextureType.h"
#include "Texture.h"
#include "ModelType.h"
#include "ModelManager.h"
#include "TextureManager.h"
#include "../Transform/Transform.h"

class DrawableObject {
public:

    DrawableObject(ModelType modelType, ShaderType shaderType, TextureType textureType = TextureType::Empty);

    void setID(unsigned int id) { id_ = id; }
    unsigned int getID() const { return id_; }
    void setTransform(const Transform& t) { transform = t; }
    Transform& getTransform() { return transform; }
    ShaderProgram* getShader() const { return shaderProgram; }

    void draw();

private:
    unsigned int id_ = 0;
    std::shared_ptr<Model> model;
    ShaderProgram* shaderProgram = nullptr;
    std::shared_ptr<Texture> texture; // RAII wrapper
    Transform transform;
};