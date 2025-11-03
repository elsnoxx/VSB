#version 330 core
in vec3 Normal;
in vec3 FragPos;

out vec4 FragColor;

uniform vec3 lightPos;
uniform vec3 lightColor;
uniform vec3 objectColor;

void main()
{
    // smìr svìtla
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);

    // Lambertùv kosinový zákon
    float diff = max(dot(norm, lightDir), 0.0);

    vec3 diffuse = diff * lightColor * objectColor;

    FragColor = vec4(diffuse, 1.0);
}
