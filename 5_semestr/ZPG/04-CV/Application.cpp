#pragma once
#include "Application.h"
#include "Callbacks.h"
#include "Models/sphere.h"
#include "Models/tree.h"
#include "Models/bushes.h"
#include "Models/gift.h"
#include "Models/suzi_flat.h"
#include "Models/suzi_smooth.h"
#include "Models/plain.h"
#include "Transform/Translation.h"

const char* vertex_shader =
"#version 330\n"
"layout(location=0) in vec3 vp;"
"layout(location=1) in vec3 vc;"
"uniform mat4 modelMatrix;"
"out vec3 color;"
"void main () {"
"     color = vc;"
//"     gl_Position = vec4 (vp, 1.0);"
"     gl_Position = modelMatrix * vec4(vp, 1.0);"
"}";

const char* fragment_shader =
"#version 330\n"
"out vec4 frag_colour;"
"in vec3 color;"
"void main () {"
"     frag_colour = vec4 (color, 1.0);"
"}";

void Application::printVersionInfo() {
    // version info
    printf("--------------------------------------------------------------------------------\n");
    printf("vendor: %s\n", glGetString(GL_VENDOR));
    printf("renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("using GLEW %s\n", glewGetString(GLEW_VERSION));

    int major, minor, revision;
    glfwGetVersion(&major, &minor, &revision);
    printf("using GLFW %i.%i.%i\n", major, minor, revision);
    printf("--------------------------------------------------------------------------------\n");
}

void Application::updateViewport() {
    glfwGetFramebufferSize(window, &width, &height);
    float ratio = width / (float)height;
    glViewport(0, 0, width, height);
}


void Application::createModels() {
    int vertexCount = sizeof(sphere) / (6 * sizeof(float)); // 6 floatù na vrchol
    model = new Model(sphere, sizeof(sphere), vertexCount);
}

void Application::createShaders() {
    VertexShader vertex(vertex_shader);
    FragmentShader fragment(fragment_shader);
    shaderProgram = new ShaderProgram(vertex, fragment);
}

void Application::createDrawableObjects() {
    drawableObject = new DrawableObject(model, shaderProgram);

    Transform t;
    t.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
    // mùžeš pøidat i další transformace:
    // t.addTransform(std::make_shared<Rotation>(45.0f, glm::vec3(0,1,0)));
    drawableObject->setTransform(t);
}

void Application::createScenes() {
    // --- Scéna 1 – koule ---
    Scene* scene1 = new Scene();
    Model* sphereModel = new Model(sphere, sizeof(sphere), sizeof(sphere) / (6 * sizeof(float)));
    ShaderProgram* shader = shaderProgram; // používáme už vytvoøený shader
    DrawableObject* obj1 = new DrawableObject(sphereModel, shader);

    Transform t1;
    t1.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, 0.0f, 0.0f)));
    obj1->setTransform(t1);

    scene1->addObject(obj1);
    scenes.push_back(scene1);

    // --- Scéna 2 – kostka (nebo nìco jiného) ---
    Scene* scene2 = new Scene();

    // zde pøedpokládám, že máš pole cube[] s vrcholy kostky
    Model* treeModel = new Model(tree, sizeof(tree), sizeof(tree) / (6 * sizeof(float)));

    DrawableObject* obj2 = new DrawableObject(treeModel, shader);

    Transform t2;
    t2.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));
    t2.addTransform(std::make_shared<Translation>(glm::vec3(-0.5f, -1.5f, 0.0f)));
    obj2->setTransform(t2);

    scene2->addObject(obj2);
    scenes.push_back(scene2);

    // --- Scéna 3 – rotující trojúhelník ---
    Scene* scene3 = new Scene();

    // jednoduchý trojúhelník
    /*float triangle[] = {
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
       -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
        0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
        -0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
       -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f
    };*/
    float triangle[] = {
         0.0f,  0.5f, 0.0f, 1.0f, 0.0f, 0.0f,  // nahoøe
         0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f,  // vpravo dole
        -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f   // vlevo dole
    };

    Model* triangleModel = new Model(triangle, sizeof(triangle), sizeof(triangle) / (6 * sizeof(float)));
    DrawableObject* obj3 = new DrawableObject(triangleModel, shaderProgram);

    // Transformace s rotací
    Transform t3;
    t3.addTransform(std::make_shared<Translation>(glm::vec3(-0.25f, 0.17f, 0.0f)));
    t3.addTransform(std::make_shared<Rotation>([&]() -> float {
        return (float)glfwGetTime() * 50.0f; // rotace podle èasu
        }, glm::vec3(0.0f, 0.0f, 1.0f))); // rotace kolem Z

    obj3->setTransform(t3);
    scene3->addObject(obj3);
    scenes.push_back(scene3);

    // SCÉNA 4 – ètyøi koule symetricky
    Scene* scene4 = new Scene();

    float offset = 3.0f; // vzdálenost od støedu

    // Souøadnice koule [X, Y, Z]
    std::vector<glm::vec3> positions = {
        glm::vec3(offset,  0.0f, 0.0f),  // +X
        glm::vec3(-offset, 0.0f, 0.0f),  // -X
        glm::vec3(0.0f,  offset, 0.0f),  // +Y
        glm::vec3(0.0f, -offset, 0.0f)   // -Y
    };

    for (const auto& pos : positions) {
        DrawableObject* obj = new DrawableObject(sphereModel, shaderProgram);
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
        t.addTransform(std::make_shared<Translation>(pos));
        obj->setTransform(t);
        scene4->addObject(obj);
    }

    scenes.push_back(scene4);

    // --- SCÉNA 5 - komplexní scéna s rùznými modely ---
    Scene* scene5 = new Scene();

    // Vytvoøení modelù
    Model* bushModel = new Model(bushes, sizeof(bushes), sizeof(bushes) / (6 * sizeof(float)));
    Model* giftModel = new Model(gift, sizeof(gift), sizeof(gift) / (6 * sizeof(float)));
    Model* suziSmoothModel = new Model(suziSmooth, sizeof(suziSmooth), sizeof(suziSmooth) / (6 * sizeof(float)));
    Model* suziFlatModel = new Model(suziFlat, sizeof(suziFlat), sizeof(suziFlat) / (6 * sizeof(float)));
    Model* plainModel = new Model(plain, sizeof(plain), sizeof(plain) / (6 * sizeof(float)));

    // 1. Vytvoøení podlahy (plain)
    {
        DrawableObject* floor = new DrawableObject(plainModel, shader);
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(5.0f, 1.0f, 5.0f)));
        t.addTransform(std::make_shared<Translation>(glm::vec3(0.0f, -2.0f, 0.0f)));
        floor->setTransform(t);
        scene5->addObject(floor);
    }

    // 2. Pøidání stromù do scény (4 stromy v rozích)
    std::vector<glm::vec3> treePositions = {
        glm::vec3(-4.0f, -3.0f, -3.0f),
        glm::vec3(3.0f, -3.0f, -3.0f),
        glm::vec3(-3.0f, -3.0f,  3.0f),
        glm::vec3(4.0f, -3.0f,  3.0f)
    };

    for (const auto& pos : treePositions) {
        DrawableObject* tree = new DrawableObject(treeModel, shader);
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));
        t.addTransform(std::make_shared<Translation>(pos));
        tree->setTransform(t);
        scene5->addObject(tree);
    }

    // 3. Pøidání keøù (6 keøù mezi stromy)
    float startX = -2.5f; // zaèátek øady
    float zPos = -4.5f;   // konstantní pozice Z
    float spacing = 1.0f; // vzdálenost mezi keøi

    for (int i = 0; i < 6; i++) {
        DrawableObject* bush = new DrawableObject(bushModel, shader);
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.2f)));

        float x = startX + i * spacing; // postupnì posouváme podle i
        t.addTransform(std::make_shared<Translation>(
            glm::vec3(x, -4.5f, zPos)
        ));

        bush->setTransform(t);
        scene5->addObject(bush);
    }

    // 4. Rotující dárky ve støedu (4 dárky)
    for (int i = 0; i < 2; i++) {
        DrawableObject* gift = new DrawableObject(giftModel, shader);
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));

        float spacing = 0.5f; // vzdálenost mezi dárky
        float x = (i % 2 == 0) ? -spacing : spacing;
        float z = (i / 2 == 0) ? -spacing : spacing;
        t.addTransform(std::make_shared<Translation>(glm::vec3(x, -2.0f, z)));

        t.addTransform(std::make_shared<Rotation>([i]() {
            return (float)glfwGetTime() * 30.0f + i * 90.0f;
            }, glm::vec3(0, 1, 0)));
        gift->setTransform(t);
        scene5->addObject(gift);
    }

    for (int i = 0; i < 2; i++) {
        DrawableObject* gift = new DrawableObject(giftModel, shader);
        Transform t;
        t.addTransform(std::make_shared<Scale>(glm::vec3(0.15f)));

        float spacing = 0.5f; // vzdálenost mezi dárky
        float x = (i % 2 == 0) ? -spacing : spacing;
        float z = (i / 2 == 0) ? -spacing : spacing;
        t.addTransform(std::make_shared<Translation>(glm::vec3(x, -3.0f, z)));

        t.addTransform(std::make_shared<Rotation>([i]() {
            return (float)glfwGetTime() * 30.0f + i * 90.0f;
            }, glm::vec3(0, 1, 0)));
        gift->setTransform(t);
        scene5->addObject(gift);
    }

    // 5. Suzi hlavy (3 øady po 2 hlavách, støídavì smooth a flat)
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 2; col++) {
            DrawableObject* suzi = new DrawableObject(
                (row % 2 == 0) ? suziSmoothModel : suziFlatModel,
                shader
            );
            Transform t;
            t.addTransform(std::make_shared<Scale>(glm::vec3(0.1f)));
            t.addTransform(std::make_shared<Translation>(
                glm::vec3(
                    col * 3.0f - 1.5f, // vìtší rozestup na X
                    0.5f + row * 2.0f, // vìtší rozestup na Y
                    0.0f
                )
            ));
            t.addTransform(std::make_shared<Rotation>([=]() {
                return (float)glfwGetTime() * 20.0f + (row * 2 + col) * 45.0f;
                }, glm::vec3(0, 1, 0)));
            suzi->setTransform(t);
            scene5->addObject(suzi);
        }
    }

    scenes.push_back(scene5);


}


void Application::switchScene(int index) {
    if (scenes.empty()) {
        printf("Error: No scenes available\n");
        return;
    }

    if (index >= 0 && index < scenes.size()) {
        printf("Switching to scene %d\n", index);
        currentSceneIndex = index;
    }
    else {
        printf("Error: Invalid scene index %d (valid range: 0-%zu)\n",
            index, scenes.size() - 1);
    }
}


//void Application::run() {
//    glEnable(GL_DEPTH_TEST);
//    while (!glfwWindowShouldClose(window)) {
//        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//        drawableObject->draw(); // vše se dìje uvnitø draw()
//        glfwPollEvents();
//        glfwSwapBuffers(window);
//    }
//}

void Application::run() {
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        if (!scenes.empty()) {
            scenes[currentSceneIndex]->draw();
        }

        glfwPollEvents();
        glfwSwapBuffers(window);
    }
}

void Application::initialization() {
    glfwSetErrorCallback(callbackError);
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        exit(EXIT_FAILURE);
    }

    // Volitelné: nastavení verze OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(800, 600, "ZPG", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetWindowUserPointer(window, this);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    // Nastavení callbackù
    glfwSetKeyCallback(window, callbackKey);
    glfwSetCursorPosCallback(window, callbackCursor);
    glfwSetMouseButtonCallback(window, callbackButton);
    glfwSetWindowFocusCallback(window, callbackWindowFocus);
    glfwSetWindowIconifyCallback(window, callbackWindowIconify);
    glfwSetWindowSizeCallback(window, callbackWindowSize);

    printVersionInfo();
    updateViewport();
}