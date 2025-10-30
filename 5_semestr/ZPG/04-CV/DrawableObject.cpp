#include "DrawableObject.h"

void DrawableObject::draw() {
    shader->use();
    glm::mat4 modelMatrix = transform.getMatrix();
    shader->setUniformMat4("modelMatrix", transform.getMatrix());
    model->draw(); // volá vlastní draw metodu modelu, která ví, jak použít VAO/VBO
}
