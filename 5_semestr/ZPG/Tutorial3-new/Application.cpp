#pragma once
#include "Application.h"

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

void Application::updateViewport(double x, double y) {
    glfwGetFramebufferSize(window, &width, &height);
    float ratio = width / (float)height;
    glViewport(0, 0, width, height);

    Scene* cur = screenManager.getCurrentScene();
    if (cur) {
        cur->getCamera()->updateScreenSize(width, height);
        cur->bindCameraAndLightToUsedShaders();
        input.onMouseMove(x, y);
    }
}

void Application::switchScene(int index) {
    screenManager.switchTo(index);
}

void Application::run() {
    glEnable(GL_DEPTH_TEST);
    while (!glfwWindowShouldClose(window)) {
        double current = glfwGetTime();
        static double last = current;
        float dt = float(current - last);
        last = current;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        screenManager.update(dt, input);
        screenManager.draw();

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

    window = glfwCreateWindow(Config::WindowWidth, Config::WindowHeight, Config::Title, NULL, NULL);
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

	// screen init
    screenManager.init();

    // Nastaven� callback�
    glfwSetKeyCallback(window, callbackKey);
    glfwSetCursorPosCallback(window, callbackCursor);
    glfwSetMouseButtonCallback(window, callbackButton);
    glfwSetWindowFocusCallback(window, callbackWindowFocus);
    glfwSetWindowIconifyCallback(window, callbackWindowIconify);
    glfwSetWindowSizeCallback(window, callbackWindowSize);

    printVersionInfo();
    updateViewport(0,0);
}