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
#include "../Transform/TransformNode.cpp"
#include "MaterialType.h"
#include "MaterialManager.h"
#include "MaterialData.h"

class DrawableObject {
public:

    DrawableObject(ModelType modelType, ShaderType shaderType);
    // back-compat constructor that accepts TextureType
    DrawableObject(ModelType modelType, ShaderType shaderType, TextureType texType);

    void setID(unsigned int id) { id_ = id; }
    unsigned int getID() const { return id_; }
    void setTransform(const Transform& t) { transform = t; }
    Transform& getTransform() { return transform; }
    ShaderProgram* getShader() const { return shaderProgram; }

    void addTexture(const std::shared_ptr<Texture>& tex) { if (tex) textures.push_back(tex); }
    void clearTextures() { textures.clear(); }

    // Optional material: set or clear. If set, material uniforms will be uploaded to shaders.
    void setMaterial(MaterialType mt) { material = MaterialManager::instance().get(mt); }
    void clearMaterial() { material = nullptr; }

    // new API: set shared transform node (composite)
    void setTransformNode(const std::shared_ptr<TransformNode>& node) {
        m_transformNode = node;
    }


    void draw();

private:
    unsigned int id_ = 0;
    std::shared_ptr<Model> model;
    ShaderProgram* shaderProgram = nullptr;
    std::vector<std::shared_ptr<Texture>> textures;
    Transform transform;
    std::shared_ptr<TransformNode> m_transformNode;
    const MaterialData* material = nullptr; // optional, not owned
};