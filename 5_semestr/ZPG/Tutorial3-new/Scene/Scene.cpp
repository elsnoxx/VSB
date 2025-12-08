#include "Scene.h"
#include "../Light/PointLight.h"
#include "../Light/SpotLight.h"
#include "../Light/DirectionalLight.h"
#include <unordered_set>

Scene::Scene() {
    camera = new Camera(glm::vec3(0.f, 1.f, 5.f));
    lightManager = new LightManager();

    headLight = new HeadLight(camera);
    headLight->intensity = 5.0f;
    lightManager->addLight(headLight);
}

Scene::~Scene() {
    delete headLight;
    delete lightManager;
    delete camera;
}

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

Light* Scene::addLight(Light* light) {
    if (!light || !lightManager) return nullptr;
    return lightManager->addLight(light);
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
    if (input.isKeyDown(GLFW_KEY_W)) { 
		std::cout << "Moving forward W\n";
        cam->forward(camSpeed); 
    }
    if (input.isKeyDown(GLFW_KEY_S)) { 
		std::cout << "Moving backward S\n";
        cam->backward(camSpeed); 
    }
    if (input.isKeyDown(GLFW_KEY_A)) { 
		std::cout << "Moving left A\n";
        cam->left(camSpeed); 
    }
    if (input.isKeyDown(GLFW_KEY_D)) {
		std::cout << "Moving right D\n";
        cam->right(camSpeed); 
    }

    if (input.isKeyPressed(GLFW_KEY_F)) {
	    std:cout << "Toggling headlight F\n";
        switchHeadLight();
    }

    // rmouse rotate
    glm::vec2 delta = input.getMouseDeltaAndReset(dt);
    if (delta.x != 0.0f || delta.y != 0.0f) {
        cam->updateOrientation(delta, dt);
    }
}

void Scene::bindCameraAndLightToUsedShaders()
{
    if (!lightManager) return;

    const int MAX_SHADER_LIGHTS = 16;
    int totalLights = lightManager->getLightsAmount();
    int n = std::min(totalLights, MAX_SHADER_LIGHTS);

    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> colors;
    std::vector<float> intensities;
    std::vector<glm::vec3> directions;
    std::vector<float> cutOffs;
    std::vector<float> outerCutOffs;
    std::vector<int> types;

    positions.reserve(n); colors.reserve(n); intensities.reserve(n);
    directions.reserve(n); cutOffs.reserve(n); outerCutOffs.reserve(n); types.reserve(n);

    for (int i = 0; i < n; ++i) {
        Light* L = lightManager->getLight(i);
        if (!L) {
            positions.push_back(glm::vec3(0.0f));
            colors.push_back(glm::vec3(0.0f));
            intensities.push_back(0.0f);
            directions.push_back(glm::vec3(0.0f));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
            types.push_back(1); // treat missing as point with zero intensity
            continue;
        }

        colors.push_back(L->color);
        // respect on/off
        intensities.push_back(L->isOn ? L->intensity : 0.0f);
        types.push_back(static_cast<int>(L->type));

        if (auto pl = dynamic_cast<PointLight*>(L)) {
            positions.push_back(pl->position);
            directions.push_back(glm::vec3(0.0f));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
        }
        else if (auto sl = dynamic_cast<SpotLight*>(L)) {
            positions.push_back(sl->position);
            directions.push_back(glm::normalize(sl->direction));
            cutOffs.push_back(sl->cutOff);
            outerCutOffs.push_back(sl->outerCutOff);
        }
        else if (auto dl = dynamic_cast<DirectionalLight*>(L)) {
            positions.push_back(-dl->direction * 10000.0f);
            directions.push_back(glm::normalize(dl->direction));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
        }
        else {
            positions.push_back(glm::vec3(0.0f));
            directions.push_back(glm::vec3(0.0f));
            cutOffs.push_back(0.0f);
            outerCutOffs.push_back(0.0f);
        }
    }

    std::unordered_set<ShaderProgram*> processed;
    for (auto* obj : objects)
    {
        ShaderProgram* shader = obj->getShader();
        if (!shader) continue;
        if (processed.count(shader)) continue;
        processed.insert(shader);

        shader->use();
        shader->setUniform("numLights", n);

        for (int i = 0; i < n; ++i) {
            shader->setUniform(("lightTypes[" + std::to_string(i) + "]").c_str(), types[i]);
            shader->setUniform(("lightPositions[" + std::to_string(i) + "]").c_str(), positions[i]);
            shader->setUniform(("lightDirections[" + std::to_string(i) + "]").c_str(), directions[i]);
            shader->setUniform(("lightColors[" + std::to_string(i) + "]").c_str(), colors[i]);
            shader->setUniform(("lightIntensities[" + std::to_string(i) + "]").c_str(), intensities[i]);
            shader->setUniform(("lightCutOffs[" + std::to_string(i) + "]").c_str(), cutOffs[i]);
            shader->setUniform(("lightOuterCutOffs[" + std::to_string(i) + "]").c_str(), outerCutOffs[i]);
        }

        shader->setUniform("viewPosition", camera->getPosition());
        shader->setUniform("shininess", 64.0f);
        shader->setUniform("ambientStrength", 0.15f);

        glUseProgram(0);
    }
}

void Scene::switchHeadLight() {
    if (!headLight) return;

    headLight->isOn = !headLight->isOn;

    if (headLight->isOn) {
        std::cout << "Headlight turned ON\n";
    }
    else {
        std::cout << "Headlight turned OFF\n";
    }

    headLight->notify(ObservableSubjects::SLight);
}