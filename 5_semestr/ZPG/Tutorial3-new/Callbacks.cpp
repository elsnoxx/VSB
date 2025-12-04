#include "callbacks.h"

// standard c++ libraries
#include <stdio.h>

// --- callbacks ---------------------------------------------------------------
void callbackError(int t_error, const char* t_description) {
	fputs(t_description, stderr);
}

void callbackWindowFocus(GLFWwindow* t_window, int t_focused) {
	printf("<callback> window focus\n");
}

void callbackWindowIconify(GLFWwindow* t_window, int t_iconified) {
	printf("<callback> window iconify\n");
}

void callbackWindowSize(GLFWwindow* window, int width, int height) {
	printf("resize %d, %d \n", width, height);
	glViewport(0, 0, width, height);
}

void callbackKey(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    app->input.onKey(key, action);

    // Přepínání scén
    if (action == GLFW_PRESS) {
        if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9) {
            int index = key - GLFW_KEY_1;
            app->switchScene(index);
        }
    }


    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


void callbackButton(GLFWwindow* window, int t_button, int t_action, int t_mode) {
	if (t_action == GLFW_PRESS)
		printf("<callback> button : button %d, action %d, mode %d\n", t_button, t_action, t_mode);
}

void callbackCursor(GLFWwindow* window, double x, double y) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    app->input.onMouseMove(x, y);
}