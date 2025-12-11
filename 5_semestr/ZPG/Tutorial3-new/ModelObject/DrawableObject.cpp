#include "DrawableObject.h"
#include <iostream>
#include <GL/glew.h>

// back-compat constructor: accepts raw Model* and wraps it as a non-owning shared_ptr
DrawableObject::DrawableObject(ModelType modelType, ShaderType shaderType)
{
    // create a non-owning shared_ptr with a no-op deleter — ModelManager still owns the model
	modelType = modelType;
    model = ModelManager::instance().get(modelType);
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
    

    glm::mat4 modelMat = this->getTransform().getMatrix();
    if (m_transformNode) {
        modelMat = m_transformNode->computeWorldMatrix();
    }
    else {
        modelMat = transform.getMatrix();
    }
    shaderProgram->use();
    // send transform to the shader (assuming shaderProgram has setUniform for mat4)
    shaderProgram->setUniform("modelMatrix", modelMat);

    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMat)));
    shaderProgram->setUniform("normalMatrix", normalMat);

    // If an optional material is assigned, upload material properties to the shader.
    // Shaders are expected to use uniforms: material.ambient, material.diffuse, material.specular, material.shininess
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

    // bind exactly ONE texture into texture unit 0 and set sampler uniform "textureUnitID"
    int texIndexToUse = 0;
    if (activeTextureIndex >= 0) texIndexToUse = activeTextureIndex;
    else texIndexToUse = 0; // default

    if (textures.size() > 0 && texIndexToUse >= 0 && texIndexToUse < (int)textures.size() && textures[texIndexToUse]) {
        auto tex = textures[texIndexToUse];
        GLuint tid = tex->getId();

        if (glIsTexture(tid)) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tid);
            shaderProgram->setUniform("textureUnitID", 0); // sampler2D
            shaderProgram->setUniform("useTexture", 1);
        }
        else {
            shaderProgram->setUniform("useTexture", 0);
        }
    }
    else {
        shaderProgram->setUniform("useTexture", 0);
    }

    // draw the model
    if (model) model->draw();

    // unbind textures (better state cleanliness)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}