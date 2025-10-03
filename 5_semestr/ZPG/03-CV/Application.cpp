#pragma once
#include "Application.h"
#include "Callbacks.h"
#include "Models/sphere.h"

//float points[] = {
//    0.0f, 0.5f, 0.0f,
//    0.5f, -0.5f, 0.0f,
//    -0.5f, -0.5f, 0.0f
//};

//float points[] = {
//    0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
//    0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
//   -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
//    0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
//    -0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f,
//   -0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f
//};

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
    glGenBuffers(1, &VBO);
    // vytvoreni buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // nahrajese data na grafickou  kartu
    //glBufferData(GL_ARRAY_BUFFER, sizeof(points), points, GL_STATIC_DRAW);
    glBufferData(GL_ARRAY_BUFFER, sizeof(sphere), sphere, GL_STATIC_DRAW);

    glGenVertexArrays(1, &VAO);
    // pracuj s nastaveny polem
    glBindVertexArray(VAO);
    // na kolik casti je to pole rozsekane
    glEnableVertexAttribArray(0);
    // je potreba zapnout shadery
    glEnableVertexAttribArray(1);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    // 3*sizeof(float) - nastavuje velikost jen pro fragmenty, pokud je potreba i normala barvy tak je potreba to zmenit na 6
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (GLvoid*)(3 * sizeof(float)));
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
    glEnable(GL_DEPTH_TEST);
    GLint status;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &status);
    if (status == GL_FALSE)
    {
        GLint infoLogLength;
        glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &infoLogLength);
        GLchar* strInfoLog = new GLchar[infoLogLength + 1];
        glGetProgramInfoLog(shaderProgram, infoLogLength, NULL, strInfoLog);
        fprintf(stderr, "Linker failure: %s\n", strInfoLog);
        delete[] strInfoLog;
    }
    while (!glfwWindowShouldClose(window)) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawArrays(GL_TRIANGLES, 0, sizeof(sphere) * 3);
        glfwPollEvents();
        // vymeni buffer, jeden se vykresli a druhy se smaze do nej se zakresli a potom se zase swapne
        glfwSwapBuffers(window);
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    exit(EXIT_SUCCESS);
}
