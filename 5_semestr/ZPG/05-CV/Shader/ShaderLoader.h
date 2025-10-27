#pragma once
//GLEW - The OpenGL Extension Wrangler Library
#include <GL/glew.h>

#include <string.h> 
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>

class ShaderLoader
{
	string loadFile(const char* fname);
};

