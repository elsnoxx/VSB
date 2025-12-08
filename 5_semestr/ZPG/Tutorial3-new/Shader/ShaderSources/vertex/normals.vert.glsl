#version 330 core

layout(location=0) in vec3 vp;
layout(location=1) in vec3 vc;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 normalVS;

void main()
{
    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(vp, 1.0);

    mat3 normalMatrix = transpose(inverse(mat3(modelMatrix)));
    vec3 normalWS = normalize(normalMatrix * vc);
    normalVS = normalize(mat3(viewMatrix) * normalWS);
}