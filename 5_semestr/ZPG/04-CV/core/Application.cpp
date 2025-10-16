#include "Application.h"
#include "../general/Callbacks.h"
#include "../Models/sphere.h"
#include "Shader/Shader.h"
#include "Shader/ShaderProgram.h"
#include "Model.h"
#include "DrawableObject.h"

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
}

void Application::createModels() {
    // Vytvoření Model objektu místo ručního VBO/VAO
    int vertexCount = sizeof(sphere) / (6 * sizeof(float)); // 6 floatů na vrchol (pozice + barva)
    model = new Model(sphere, sizeof(sphere), vertexCount);
}

void Application::createShaders() {
    // Vytvoření Shader objektu (obsahuje VS a FS)
    Shader* shader = new Shader(vertex_shader, fragment_shader);
    
    // Vytvoření ShaderProgram objektu
    shaderProgram = new ShaderProgram(shader);
}

void Application::createDrawableObjects() {
    // Vytvoření DrawableObject
    drawableObject = new DrawableObject(model, shaderProgram);
}

void Application::initialization() {
    glfwSetErrorCallback(callbackError);
    if (!glfwInit()) {
        fprintf(stderr, "ERROR: could not start GLFW3\n");
        exit(EXIT_FAILURE);
    }

    // Volitelně: nastavení verze OpenGL
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(800, 600, "ZPG", NULL, NULL);
    if (!window) {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    // Nastavení callbacků
    glfwSetKeyCallback(window, callbackKey);
    glfwSetCursorPosCallback(window, callbackCursor);
    glfwSetMouseButtonCallback(window, callbackButton);
    glfwSetWindowFocusCallback(window, callbackWindowFocus);
    glfwSetWindowIconifyCallback(window, callbackWindowIconify);
    glfwSetWindowSizeCallback(window, callbackWindowSize);

    printVersionInfo();
    updateViewport();
    
    // Vytvoření objektů podle nové struktury
    createShaders();
    createModels();
    createDrawableObjects();
}

void Application::run() {
    glEnable(GL_DEPTH_TEST);
    
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // Vykreslení pomocí DrawableObject
        drawableObject->draw();
        
        glfwPollEvents();
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    delete drawableObject;
    delete model;
    delete shaderProgram;
    
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}