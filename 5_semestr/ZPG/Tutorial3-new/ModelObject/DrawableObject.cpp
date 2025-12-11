#include "DrawableObject.h"
#include <iostream>
#include <GL/glew.h>

// DrawableObject: wrapper for a renderable model + shader + optional textures/material.
// - Holds a non-owning shared_ptr to `Model` provided by `ModelManager`.
// - `material` is an optional non-owning pointer to `MaterialData`.
// - Textures are stored as shared_ptrs (cached/shared by TextureManager).
DrawableObject::DrawableObject(ModelType modelType, ShaderType shaderType)
{
    // store requested model type
    this->modelType = modelType;

    // lookup (non-owning) shared_ptr to model - ModelManager retains ownership
    model = ModelManager::instance().get(modelType);

    // fetch shader program pointer from factory (not owned here)
    shaderProgram = ShaderFactory::Get(shaderType);
}

// new constructor that also loads a texture from TextureType
DrawableObject::DrawableObject(ModelType modelType, ShaderType shaderType, TextureType texType)
    : DrawableObject(modelType, shaderType)
{
    if (texType != TextureType::Empty) {
        auto tex = TextureManager::instance().get(texType);
        if (tex) textures.push_back(tex);
    }
}


void DrawableObject::draw() {
    if (!shaderProgram) {
        std::cerr << "DrawableObject::draw() ERROR Shader not set!\n";
        return;
    }
    

    // Compute model/world matrix. Prefer the attached transform node's world
    // matrix if present (composite transforms), otherwise use local transform.
    glm::mat4 modelMat = this->getTransform().getMatrix();
    if (m_transformNode) {
        modelMat = m_transformNode->computeWorldMatrix();
    }
    else {
        modelMat = transform.getMatrix();
    }

    // Activate shader and upload matrices
    shaderProgram->use();
    shaderProgram->setUniform("modelMatrix", modelMat);

    // Normal matrix = inverse-transpose of model's 3x3 part
    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMat)));
    shaderProgram->setUniform("normalMatrix", normalMat);

    // Upload material properties if provided. Uniform names must match the
    // shader's expectations; this project uses the names below by default.
    if (material) {
        shaderProgram->setUniform("materialDiffuse", material->diffuse);
        shaderProgram->setUniform("materialSpecular", material->specular);
        shaderProgram->setUniform("shininess", material->shininess);
        shaderProgram->setUniform("ambientStrength", 0.1f);
        shaderProgram->setUniform("useMaterial", 1);
    }
    else {
        shaderProgram->setUniform("useMaterial", 0);
    }

    // Bind a single texture to unit 0. If no texture is available, tell the
    // shader to skip texturing via `useTexture` uniform.
    int texIndexToUse = 0;
    if (activeTextureIndex >= 0) texIndexToUse = activeTextureIndex;

    if (textures.size() > 0 && texIndexToUse >= 0 && texIndexToUse < (int)textures.size() && textures[texIndexToUse]) {
        auto tex = textures[texIndexToUse];
        GLuint tid = tex->getId();

        if (glIsTexture(tid)) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tid);
            shaderProgram->setUniform("textureUnitID", 0); // sampler2D expects unit index
            shaderProgram->setUniform("useTexture", 1);
        }
        else {
            shaderProgram->setUniform("useTexture", 0);
        }
    }
    else {
        shaderProgram->setUniform("useTexture", 0);
    }

    // Draw the model (Model::draw binds VAO/attributes internally)
    if (model) model->draw();

    // Reset texture/program state to keep GL state predictable for other code
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    // Debug: check for GL errors
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}