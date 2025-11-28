#include "Scene.h"
Scene::Scene(ShaderProgram* shader) : shaderProgram(shader)
{
    camera = new Camera(glm::vec3(0, 0, 5));
    camera->addObserver(shaderProgram);

}

