#include "DrawableObject.h"
#include <iostream>
#include <GL/glew.h>

// back-compat konstruktor: přijme raw Model* a zabalí ho jako non-owning shared_ptr
DrawableObject::DrawableObject(ModelType modelType, ShaderType shaderType)
{
    // vytvoříme non-owning shared_ptr s no-op deleter — ModelManager stále vlastní model
    model = ModelManager::instance().get(modelType);
    shaderProgram = ShaderFactory::Get(shaderType);
}

// nový konstruktor, který zároveň načte texturu z TextureType
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

    // pošli transform do shaderu (předpokládáme že shaderProgram má setUniform pro mat4)
    shaderProgram->setUniform("modelMatrix", modelMat);

    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMat)));
    shaderProgram->setUniform("normalMatrix", normalMat);

    // bind exactly ONE texture into texture unit 0 and set sampler uniform "textureUnitID"
    if (textures.size() > 0 && textures[0]) {
        auto tex = textures[0];
        GLuint tid = tex->getId(); // adjust if your API uses different getter
        if (glIsTexture(tid)) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, tid);
            shaderProgram->setUniform("textureUnitID", 0); // set INT = texture unit
            shaderProgram->setUniform("useTexture", 1);
        }
        else {
            shaderProgram->setUniform("useTexture", 0);
            printf("[Drawable] texture id invalid: %u\n", tid);
        }
    }
    else {
        shaderProgram->setUniform("useTexture", 0);
    }

    // vykresli model
    if (model) model->draw();

    // unbind textures (lepší čistota stavu)
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}