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
#include "./Scene/Camera.h"
#include "Shader/ShaderLoader.h"

const char* vertex_shader =
"#version 330\n"
"layout(location=0) in vec3 vp;"
"layout(location=1) in vec3 vc;"
"uniform mat4 modelMatrix;"
"out vec3 color;"
"void main () {"
"     color = vc;"
"     gl_Position = vec4 (vp, 1.0);"
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
    if (camera) camera->setAspect(ratio);
}


void Application::createModels() {
    int vertexCount = sizeof(sphere) / (6 * sizeof(float)); // 6 float� na vrchol
    model = new Model(sphere, sizeof(sphere), vertexCount);
}

void Application::createShaders() {
    // Načti shadery ze souborů místo hardcoded stringů
    shaderProgram = ShaderLoader::createShaderProgram(
        "Assets/Shader/vertex/Base.vert",
        "Assets/Shader/fragment/Base.frag"
    );

    if (!shaderProgram) {
        std::cerr << "Failed to create shader program!" << std::endl;
        return;
    }

    // propojíme shader s kamerou (observer)
    if (camera) shaderProgram->addCamera(camera);
}

void Application::createScenes() {
    // Příklad vytváření různých shaderů pro různé scény
    ShaderProgram* lambertShader = ShaderLoader::createShaderProgram(
        "Assets/Shader/vertex/Base.vert",
        "Assets/Shader/fragment/lambert.frag"
    );
    
    ShaderProgram* phongShader = ShaderLoader::createShaderProgram(
        "Assets/Shader/vertex/Base.vert", 
        "Assets/Shader/fragment/phong.frag"
    );
    
    // Připoj kameru k oběma shaderům
    if (camera) {
        if (lambertShader) lambertShader->addCamera(camera);
        if (phongShader) phongShader->addCamera(camera);
    }
    
    // Vytvoř scény s různými shadery...

    // SCÉNA 1 - čtyři koule symetricky
    Scene* scene1 = new Scene();

    float offset = 0.5f; // vzdálenost od středu

    // Souřadnice koule [X, Y, Z]
    std::vector<glm::vec3> positions = {
        glm::vec3(offset,  offset, 0.0f),
        glm::vec3(-offset,  offset, 0.0f),
        glm::vec3(offset, -offset, 0.0f),
        glm::vec3(-offset, -offset, 0.0f)
    };

    for (const auto& pos : positions) {
        DrawableObject* obj = new DrawableObject(model, shaderProgram);
        Transform t;
        t.addTransform(std::make_shared<Translation>(pos));
        obj->setTransform(t);
        scene1->addObject(obj);
    }

    scenes.push_back(scene1);
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
//        drawableObject->draw(); // v�e se d�je uvnit� draw()
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

    // Voliteln�: nastaven� verze OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(800, 600, "ZPG", NULL, NULL);
    if (!window) {
        glfwTerminate();
    camera->setAspect(width / (float)height);
    camera->setFOV(60.0f);
    camera->setNearFar(0.1f, 100.0f);

    printVersionInfo();
    updateViewport();
}

// při změně velikosti okna aktualizujeme projekční poměr kamery
void Application::updateViewport() {
    glfwGetFramebufferSize(window, &width, &height);
    float ratio = width / (float)height;
    glViewport(0, 0, width, height);
    if (camera) camera->setAspect(ratio);    glfwSetWindowUserPointer(window, this);
}
rrent(window);
void Application::createShaders() {   glfwSwapInterval(1);
    VertexShader vertex(vertex_shader);
    FragmentShader fragment(fragment_shader);
    shaderProgram = new ShaderProgram(vertex, fragment);

    // propojíme shader s kamerou (observer)
    if (camera) shaderProgram->setCamera(camera);width, &height);
}

// callback kurzoru volá tuto metodu    // Nastaven� callback�
void Application::processCursor(double x, double y) {backKey);
    if (camera) camera->processMouseMovement(x, y);llbackCursor);
}    printVersionInfo();
    updateViewport();
}

// při změně velikosti okna aktualizujeme projekční poměr kamery
void Application::updateViewport() {
    glfwGetFramebufferSize(window, &width, &height);
    float ratio = width / (float)height;
    glViewport(0, 0, width, height);
    if (camera) camera->setAspect(ratio);
}

void Application::createShaders() {
    VertexShader vertex(vertex_shader);
    FragmentShader fragment(fragment_shader);
    shaderProgram = new ShaderProgram(vertex, fragment);

    // propojíme shader s kamerou (observer)
    if (camera) shaderProgram->setCamera(camera);
}

// callback kurzoru volá tuto metodu
void Application::processCursor(double x, double y) {
    if (camera) camera->processMouseMovement(x, y);
}