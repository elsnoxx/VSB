#include "Scene.h"

Camera* Scene::getCamera() {
	return camera;
}

void Scene::addObject(DrawableObject* obj) {
    objects.push_back(obj);

    ShaderProgram* shader = obj->getShader();

    if (shader) {
        shader->attachCamera(camera);
    }
}

void Scene::draw() {
    for (auto& obj : objects) {
        obj->draw();
    }
}

void Scene::update(float dt, InputManager& input)
{
    float camSpeed = 5.0f * dt;
    auto cam = getCamera();

    // WSAD pohyb
    if (input.isKeyDown(GLFW_KEY_W))
    {
        std::cout << "W pressed\n";
        cam->forward(camSpeed);
    }

    if (input.isKeyDown(GLFW_KEY_S))
    {
        std::cout << "S pressed\n";
        cam->backward(camSpeed);
    }

    if (input.isKeyDown(GLFW_KEY_A))
    {
        std::cout << "A pressed\n";
        cam->left(camSpeed);
    }
    if (input.isKeyDown(GLFW_KEY_D))
    {
        std::cout << "D pressed\n";
        cam->right(camSpeed);
    }


    // rotace myší
    glm::vec2 delta = input.getMouseDeltaAndReset(dt);
    if (delta.x != 0.0f || delta.y != 0.0f) {
        cam->updateOrientation(delta, dt);
    }
}

void Scene::bindCameraAndLightToUsedShaders()
{
    for (auto* obj : objects)
    {
        ShaderProgram* shader = obj->getShader();
        if (!shader) continue;

        camera->attach(shader);
        shader->use();
        shader->setUniform("lightPosition", glm::vec3(10.0f, 10.0f, 10.0f));
        shader->setUniform("viewPosition", camera->getPosition());
        shader->setUniform("shininess", 32.0f);

    }
}
