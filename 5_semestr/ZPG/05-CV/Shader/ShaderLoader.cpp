#include "ShaderLoader.h"


ShaderLoader::loadFile(const char* fname)
{
	std::ifstream file;
	std::stringstream buf;
	std::string ret = "";
	file.open(fname, std::ios::in);
	if (file.is_open())
	{
		buf << file.rdbuf();
		ret = buf.str();
	}
	else
	{
		std::cerr << "Could not open " << fname << " for reading!" << std::endl;
	}
	file.close();
	return ret;
}
