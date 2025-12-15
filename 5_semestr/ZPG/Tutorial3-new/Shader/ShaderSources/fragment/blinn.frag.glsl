#version 450 core

in vec3 worldPosition;
in vec3 worldNormal;
in vec2 uv;

out vec4 frag_colour;

// TEXTURE
uniform sampler2D textureUnitID;
uniform int useTexture;

// LIGHTS
struct Light {
    int type;
    int isOn;
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    float cutOff;
    float outerCutOff;
};
uniform Light lights[16];
uniform int numLights;

uniform vec3 viewPosition;
uniform float shininess;
uniform float ambientStrength;
uniform vec3 materialDiffuse;
uniform vec3 materialSpecular;
uniform vec3 materialEmissive;
uniform int useEmissive;

void main()
{
    vec3 baseColor = materialDiffuse;

    if (useTexture == 1)
    {
        vec4 t = texture(textureUnitID, uv);
        baseColor = t.rgb;
    }

    vec3 N = normalize(worldNormal);
    vec3 V = normalize(viewPosition - worldPosition);

    vec3 totalDiffuse = vec3(0.0);
    vec3 totalSpec = vec3(0.0);

    int n = clamp(numLights, 0, 16);
    for (int i = 0; i < n; ++i)
    {
        Light L = lights[i];
        if (L.isOn == 0) continue;

        vec3 Li;
        float attenuation = 1.0;
        float dist = 0.0;

        if (L.type == 0) {
            // Directional light -> direction provided (no attenuation)
            Li = normalize(-L.direction); // direction points from light -> scene
        } else {
            // Point or Spot -> compute vector to light and attenuation by distance
            vec3 Lvec = L.position - worldPosition;
            dist = length(Lvec);
            Li = normalize(Lvec);
            // basic distance attenuation
            attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * dist * dist);
        }

        float diff = max(dot(N, Li), 0.0);
        vec3 R = reflect(-Li, N);
        float spec = (diff > 0.0) ? pow(max(dot(V, R), 0.0), shininess) : 0.0;

        // SPOT-specific falloff
        float spotEffect = 1.0;
        if (L.type == 2)
        {
            vec3 spotDir = normalize(L.direction);
            float theta = dot(-Li, spotDir);
            float inner = L.cutOff;
            float outer = L.outerCutOff;
            float eps = max(inner - outer, 0.0001);
            spotEffect = clamp((theta - outer) / eps, 0.0, 1.0);
        }

        vec3 lightCol = L.color * L.intensity * spotEffect;

        totalDiffuse += attenuation * lightCol * diff;
        totalSpec += attenuation * lightCol * spec;
    }

    vec3 ambient = ambientStrength * baseColor;
    vec3 diffuse = totalDiffuse * baseColor;
    vec3 specular = materialSpecular * totalSpec;

    vec3 result = ambient + diffuse + specular;

    // ----- ADD EMISSIVE -----
    if (useEmissive == 1) {
        result += materialEmissive;
    }

    frag_colour = vec4(result, 1.0);
}