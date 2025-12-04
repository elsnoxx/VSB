#include "Application.h"

int main(void)
{
	Application* app = new Application();
	app->initialization(); //OpenGL inicialization

	app->run(); //Rendering 
	
}