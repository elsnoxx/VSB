#include "Scene.h"
//#include "../ModelObject/ModelManager.h"
//#include "../Transform/Scale.h"
//#include "../Transform/Translation.h"

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
    bindCameraAndLightToUsedShaders();
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


    // rmouse rotate
    glm::vec2 delta = input.getMouseDeltaAndReset(dt);
    if (delta.x != 0.0f || delta.y != 0.0f) {
        cam->updateOrientation(delta, dt);
    }
}

void Scene::bindCameraAndLightToUsedShaders()
{
    // defalut light
    glm::vec3 lightPos = glm::vec3(0.0f, 0.0f, 1.5f);
    glm::vec3 lightCol = glm::vec3(1.0f);
    float lightIntensity = 1.0f;

    if (!lights.empty()) {
        lightPos = lights[0].position;
        lightCol = lights[0].color;
        lightIntensity = lights[0].intensity;
    }

    for (auto* obj : objects)
    {
        ShaderProgram* shader = obj->getShader();
        if (!shader) continue;

        camera->attach(shader);
        shader->use();

        shader->setUniform("lightPosition", lightPos);
        shader->setUniform("viewPosition", camera->getPosition());
        shader->setUniform("shininess", 64.0f);
        shader->setUniform("ambientStrength", 0.15f);
    }
}


//void Scene::spawnTree(const glm::vec3& pos)
//{
//    Model* treeModel = ModelManager::instance().get(ModelType::Tree);
//    DrawableObject* obj = new DrawableObject(treeModel, ShaderType::Basic);
//
//    Transform t;
//    t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
//    t.addTransform(std::make_shared<Translation>(pos));
//
//    obj->setTransform(t);
//    Scene::addObject(obj);
//}