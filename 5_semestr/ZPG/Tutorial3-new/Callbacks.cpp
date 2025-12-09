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

    if (action == GLFW_PRESS) {
        if (key >= GLFW_KEY_1 && key <= GLFW_KEY_9 ) {
            int index = key - GLFW_KEY_1;
            app->switchScene(index);
            
        }
        // FOV => F1 = 45°, F2 = 90°, F3 = 130°
            if (key == GLFW_KEY_F1) {
            app->updateFOV(glm::radians(45.0f));
            printf("[FOV] Set to 45 degrees\n");
            
        }
        else if (key == GLFW_KEY_F2) {
            app->updateFOV(glm::radians(90.0f));
            printf("[FOV] Set to 90 degrees\n");
            
        }
        else if (key == GLFW_KEY_F3) {
            app->updateFOV(glm::radians(130.0f));
            printf("[FOV] Set to 130 degrees\n");
            
        }
        
    }


    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GL_TRUE);
}


void callbackButton(GLFWwindow* window, int t_button, int t_action, int t_mode) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

    if (t_action == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        printf("<callback> button : button %d, action %d, mode %d, cursor %f %f\n", t_button, t_action, t_mode, x, y);
        app->handleMouseClick(x, y, t_button);
    }
}

void callbackCursor(GLFWwindow* window, double x, double y) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (!app) return;

	app->updateViewport(x, y);
}