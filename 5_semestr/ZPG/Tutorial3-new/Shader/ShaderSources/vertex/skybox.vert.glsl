#version 450 core
layout(location = 0) in vec3 vp;

out vec3 TexCoords;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main()
{
    TexCoords = vp;

    // Z view matrix odstraníme pozici kamery
    mat4 viewNoTranslation = mat4(mat3(viewMatrix));

    gl_Position = projectionMatrix * viewNoTranslation * vec4(vp, 1.0);
}
