#pragma once
#include "Application.h"
#include "Callbacks.h"

float points[] = {
    0.0f, 0.5f, 0.0f,
    0.5f, -0.5f, 0.0f,
    -0.5f, -0.5f, 0.0f
};

const char* vertex_shader =
"#version 330\n"
"layout(location=0) in vec3 vp;"
"void main () {"
"     gl_Position = vec4 (vp, 1.0);"
"}";




const char* fragment_shader =
"#version 330\n"
"out vec4 frag_colour;"
"void main () {"
"     frag_colour = vec4 (0.5, 0.0, 0.5, 1.0);"
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
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);
}

void Application::createShaders() {
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertex_shader, NULL);
    glCompileShader(vertexShader);

    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragment_shader, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, fragmentShader);
    glAttachShader(shaderProgram, vertexShader);
    glLinkProgram(shaderProgram);
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


void Application::run() {
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwPollEvents();
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
