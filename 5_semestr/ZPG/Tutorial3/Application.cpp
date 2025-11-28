#pragma once
#include "Application.h"
#include "Callbacks.h"
#include "Transform/Translation.h"
#include "Scene/SceneFactory.h"

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
    scenes = SceneFactory::createAllScenes(shaderProgram);
}

Camera* Application::getCurrentCamera() {
    if (scenes.empty()) return nullptr;
    return scenes[currentSceneIndex]->getCamera();
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

        shaderProgram->use();

        shaderProgram->updateMatricesInGPU();

        scenes[currentSceneIndex]->draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
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
    
    // camera settings
    camera = new Camera(glm::vec3(0, 0, 5), shaderProgram);


    // settings callbacks
    glfwSetKeyCallback(window, callbackKey);
    glfwSetCursorPosCallback(window, callbackCursor);
    glfwSetMouseButtonCallback(window, callbackButton);
    glfwSetWindowFocusCallback(window, callbackWindowFocus);
    glfwSetWindowIconifyCallback(window, callbackWindowIconify);
    glfwSetWindowSizeCallback(window, callbackWindowSize);

    printVersionInfo();
    updateViewport();
}