#version 330 core

in vec3 worldPosition;
in vec3 worldNormal;
in vec3 albedo;

out vec4 frag_colour;

uniform vec3 lightPosition;
uniform vec3 viewPosition;
uniform float shininess;

void main()
{
    vec3 N = normalize(worldNormal);
    vec3 L = normalize(lightPosition - worldPosition);
    float diff = max(dot(N, L), 0.0);

    // Blinn-Phong nebo Phong - tady klasický Phong
    vec3 V = normalize(viewPosition - worldPosition);
    vec3 R = reflect(-L, N);
    float spec = 0.0;
    if (diff > 0.0) {
        spec = pow(max(dot(V, R), 0.0), shininess);
    }

    vec3 ambient = 0.1 * albedo;
    vec3 diffuse = diff * albedo;
    vec3 specular = spec * vec3(1.0);

    vec3 color = ambient + diffuse + specular;
    frag_colour = vec4(color, 1.0);
}