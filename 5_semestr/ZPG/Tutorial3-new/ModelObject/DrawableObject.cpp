#include "DrawableObject.h"
#include "TextureLoader.h"
#include <iostream>
#include <GL/glew.h>

// helper: load texture according to global TextureType enum (TextureType.h)
static GLuint loadTextureForType(TextureType tt) {
    switch (tt) {
    case TextureType::Empty:  return 0u;
    case TextureType::Shrek:  return LoadTexture("ModelObject/textures/shrek.png");
    case TextureType::Fiona:  return LoadTexture("ModelObject/textures/fiona.png");
    case TextureType::Toilet: return LoadTexture("ModelObject/textures/toilet.png");
    default: return 0u;
    }
}

DrawableObject::DrawableObject(Model* m, ShaderType shaderType, TextureType textureType) {
    shaderProgram = ShaderFactory::Get(shaderType);
    model = m;
    textureID = loadTextureForType(textureType);
    this->textureType = textureType;
}

void DrawableObject::setTextureID(GLuint id) { textureID = id; }

GLuint DrawableObject::getTextureID() const { return textureID; }

void DrawableObject::setTextureType(TextureType t) {
    if (textureType == t) return;
    textureType = t;
    // pokud loader m� free, uvolni starou texturu zde
    textureID = loadTextureForType(t);
}

TextureType DrawableObject::getTextureType() const { return textureType; }

void DrawableObject::draw() {
    if (!shaderProgram) {
        std::cerr << "DrawableObject::draw() ERROR Shader not set!\n";
        return;
    }

    shaderProgram->use();
    
    shaderProgram->setUniform("modelMatrix", transform.getMatrix());

    if (textureID != 0u) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        shaderProgram->setUniform("textureUnitID", 0);
    }

    if (model) model->draw();

    // cleanup
    if (textureID != 0u) {
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glUseProgram(0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}