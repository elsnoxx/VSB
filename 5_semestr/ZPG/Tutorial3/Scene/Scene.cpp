#include "Scene.h"

Scene::Scene(ShaderProgram* shader)
    : shaderProgram(shader)
{
    camera = new Camera(glm::vec3(0, 0, 5), shaderProgram);
}

void Scene::addObject(DrawableObject* obj) {
    objects.push_back(obj);
}

void Scene::draw() {
    for (auto& obj : objects)
        obj->draw();
}
