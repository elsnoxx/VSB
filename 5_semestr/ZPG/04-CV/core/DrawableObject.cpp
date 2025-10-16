#include "DrawableObject.h"
#include <glm/gtc/type_ptr.hpp>

DrawableObject::DrawableObject(Model* model, ShaderProgram* shaderProgram) 
    : model(model), shaderProgram(shaderProgram), modelMatrix(1.0f) {
}

void DrawableObject::draw() const {
    shaderProgram->use();
    
    // Předání model matrix do shaderu
    shaderProgram->setUniform("modelMatrix", modelMatrix);
    
    model->draw();
}