#version 330 core

in vec3 worldPosition;
in vec3 worldNormal;
in vec3 albedo;

out vec4 frag_colour;

uniform vec3 lightPosition;

void main()
{
    vec3 N = normalize(worldNormal);
    vec3 L = normalize(lightPosition - worldPosition);
    float diff = max(dot(N, L), 0.0);

    vec3 ambient = 0.1 * albedo;
    vec3 diffuse = diff * albedo;

    frag_colour = vec4(ambient + diffuse, 1.0);
}