#include "SceneFactory.h"
#include "../Transform/Translation.h"
#include "../Transform/Rotation.h"
#include "../Transform/Scale.h"

std::vector<Scene*> SceneFactory::createAllScenes(ShaderProgram* shader) {
    return {
        createScene1(shader),
        createScene2(shader),
        createScene3(shader),
        createScene4(shader),
        createScene5(shader)
    };
}

Scene* SceneFactory::createScene1(ShaderProgram* shader) {
    Scene* scene = new Scene();

    Model* sphereModel = new Model(sphere, sizeof(sphere), sizeof(sphere) / (6 * sizeof(float)));
    DrawableObject* obj = new DrawableObject(sphereModel, shader);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(0, 0, 0)));
    obj->setTransform(t);

    scene->addObject(obj);
    return scene;
}

Scene* SceneFactory::createScene2(ShaderProgram* shader) {
    Scene* scene = new Scene();

    Model* treeModel = new Model(tree, sizeof(tree), sizeof(tree) / (6 * sizeof(float)));
    DrawableObject* obj = new DrawableObject(treeModel, shader);

    Transform t;
    t.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));
    t.addTransform(std::make_shared<Translation>(glm::vec3(-0.5f, -1.5f, 0.0f)));

    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createScene3(ShaderProgram* shader) {
    Scene* scene = new Scene();

    float triangle[] = {
         0.0f,  0.5f, 0.0f, 1, 0, 0,
         0.5f, -0.5f, 0.0f, 0, 1, 0,
        -0.5f, -0.5f, 0.0f, 0, 0, 1
    };

    Model* model = new Model(triangle, sizeof(triangle), sizeof(triangle) / (6 * sizeof(float)));
    DrawableObject* obj = new DrawableObject(model, shader);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(-0.25f, 0.17f, 0.0f)));

    t.addTransform(std::make_shared<Rotation>([]() {
        return (float)glfwGetTime() * 50.0f;
        }, glm::vec3(0, 0, 1)));

    obj->setTransform(t);
    scene->addObject(obj);

    return scene;
}

Scene* SceneFactory::createScene4(ShaderProgram* shader) {
    Scene* scene = new Scene();

    Model* sphereModel = new Model(sphere, sizeof(sphere), sizeof(sphere) / (6 * sizeof(float)));
    float offset = 3.0f;

    std::vector<glm::vec3> positions = {
        {offset,0,0}, {-offset,0,0}, {0,offset,0}, {0,-offset,0}
    };

    for (auto& pos : positions) {
        DrawableObject* obj = new DrawableObject(sphereModel, shader);

        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
        t.addTransform(std::make_shared<Translation>(pos));

        obj->setTransform(t);
        scene->addObject(obj);
    }

    return scene;
}

Scene* SceneFactory::createScene5(ShaderProgram* shader) {
    Scene* scene = new Scene();

    // pøesuneš sem svùj celý komplexní obsah scény 5
    // (stromy, keøe, suzi, dárky atd.)

    return scene;
}
