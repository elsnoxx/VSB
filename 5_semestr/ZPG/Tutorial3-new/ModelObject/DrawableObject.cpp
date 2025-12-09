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
    shaderProgram->use();

    // pošli transform do shaderu (předpokládáme že shaderProgram má setUniform pro mat4)
    shaderProgram->setUniform("modelMatrix", modelMat);

    glm::mat3 normalMat = glm::transpose(glm::inverse(glm::mat3(modelMat)));
    shaderProgram->setUniform("normalMatrix", normalMat);

    // pokud jsou nějaké textury, bindneme je do jednotek 0..N-1 a nastavíme sampler-uniformy
    for (size_t i = 0; i < textures.size(); ++i) {
        if (!textures[i] || textures[i]->getID() == 0u) continue;
        glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
        glBindTexture(GL_TEXTURE_2D, textures[i]->getID());
        // uniform jméno: texture0, texture1, ...
        std::string uname = "texture" + std::to_string(i);
        shaderProgram->setUniform(uname.c_str(), static_cast<int>(i));
    }

    // zabezpečení: pokud žádná textura, pošleme texture0=0 (většinou již nastaveno někde jinde)
    if (textures.empty()) {
        shaderProgram->setUniform("texture0", 0);
    }

    // vykresli model
    if (model) model->draw();

    // unbind textures (lepší čistota stavu)
    for (size_t i = 0; i < textures.size(); ++i) {
        glActiveTexture(GL_TEXTURE0 + static_cast<GLenum>(i));
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}