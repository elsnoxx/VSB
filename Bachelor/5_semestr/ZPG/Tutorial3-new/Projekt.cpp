#include "Application.h"

int main(void)
{
	Application* app = new Application();
	app->initialization(); // OpenGL initialization

	app->run(); //Rendering 
	
}