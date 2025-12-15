#version 330 core

layout(location = 0) in vec3 vp;
layout(location = 1) in vec3 vn;

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

    worldNormal = normalize(transpose(inverse(mat3(modelMatrix))) * vp);


    albedo = vec3(1,1,1);
    gl_Position = projectionMatrix * viewMatrix * worldPos4;
}