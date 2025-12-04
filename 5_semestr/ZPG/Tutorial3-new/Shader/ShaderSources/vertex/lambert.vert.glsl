#version 330 core

layout(location = 0) in vec3 vp;
layout(location = 1) in vec3 vc;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

out vec3 worldPosition;
out vec3 worldNormal;
out vec3 albedo;

void main()
{
    vec4 worldPos4 = modelMatrix * vec4(vp, 1.0);
    worldPosition = worldPos4.xyz;
    worldNormal = normalize(mat3(transpose(inverse(modelMatrix))) * vp);
    albedo = vc;
    gl_Position = projectionMatrix * viewMatrix * worldPos4;
}