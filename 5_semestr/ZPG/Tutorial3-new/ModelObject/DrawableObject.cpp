#include "DrawableObject.h"
#include <iostream>
#include <GL/glew.h>



// back-compat konstruktor: přijme raw Model* a zabalí ho jako non-owning shared_ptr
DrawableObject::DrawableObject(ModelType modelType, ShaderType shaderType, TextureType textureType)
{
    // vytvoříme non-owning shared_ptr s no-op deleter — ModelManager stále vlastní model
    model = ModelManager::instance().get(modelType);
    shaderProgram = ShaderFactory::Get(shaderType);
    texture = TextureManager::instance().get(textureType);
}

void DrawableObject::draw() {
    if (!shaderProgram) {
        std::cerr << "DrawableObject::draw() ERROR Shader not set!\n";
        return;
    }

    shaderProgram->use();

    // pošli transform do shaderu (předpokládáme že shaderProgram má setUniform pro mat4)
    shaderProgram->setUniform("modelMatrix", transform.getMatrix());

    // Pokud máme texturu, bindneme ji a nastavíme sampler uniformu na texture unit 0
    if (texture && texture->getID() != 0u) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture->getID());
        shaderProgram->setUniform("textureUnitID", 0);
    }
    else {
        // volitelně: nastavit sampler na 0 i bez textury (bezpečnost)
        shaderProgram->setUniform("textureUnitID", 0);
    }

    // vykresli model
    if (model) model->draw();

    // unbind texture a program
    if (texture && texture->getID() != 0u) {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}