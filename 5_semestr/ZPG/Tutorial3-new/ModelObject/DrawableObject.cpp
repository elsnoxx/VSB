#include "DrawableObject.h"

DrawableObject::DrawableObject(Model* m, ShaderType shaderType) {
	shaderProgram = ShaderFactory::Get(shaderType);
	model = m;
}

void DrawableObject::draw() {
    if (!shaderProgram) {
        std::cerr << "DrawableObject::draw() ERROR Shader not set!\n";
        return;
    }

    shaderProgram->use();
    glm::mat4 modelMatrix = transform.getMatrix();
    shaderProgram->setUniform("modelMatrix", transform.getMatrix());

    if (textureID != 0) {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);
        shaderProgram->setUniform("textureUnitID", 0);
    }

    model->draw();

    glUseProgram(0);

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        std::cerr << "GL error after DrawableObject::draw(): 0x" << std::hex << err << std::dec << "\n";
    }
}
