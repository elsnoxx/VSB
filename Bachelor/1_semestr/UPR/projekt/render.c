#include "hra.h"
#include "constants.h"

void renderPlayer(SDL_Renderer *renderer, hrac hrac, SDL_Texture *textures[])
{
    switch (hrac.smer)
    {
    case 'r':
        SDL_RenderCopy(renderer, textures[0], NULL, &hrac.poloha);
        break;
    case 'l':
        SDL_RenderCopy(renderer, textures[1], NULL, &hrac.poloha);
        break;
    case 'd':
        SDL_RenderCopy(renderer, textures[2], NULL, &hrac.poloha);
        break;
    case 'u':
        SDL_RenderCopy(renderer, textures[3], NULL, &hrac.poloha);
        break;
    default:
        break;
    }
}

void render_strelby(SDL_Renderer *renderer, hrac *hrac, SDL_Texture *bullet, SDL_Rect domek, double delta_time)
{
    if (hrac->vystrel)
    {
        switch (hrac->smer_strely)
        {
        case 'r':
            hrac->naboj.x += 400 * delta_time;
            break;
        case 'l':
            hrac->naboj.x -= 400 * delta_time;
            break;
        case 'd':
            hrac->naboj.y += 400 * delta_time;
            break;
        case 'u':
            hrac->naboj.y -= 400 * delta_time;
            break;
        }
        if (SDL_HasIntersection(&hrac->naboj, &domek))
        {
            hrac->vystrel = false;
            return;
        }

        if (hrac->naboj.y < 0 || hrac->naboj.y > WINDOW_HEIGHT || hrac->naboj.x < 0 || hrac->naboj.x > WINDOW_WIDHT)
        {
            hrac->vystrel = false;
        }
        SDL_RenderCopy(renderer, bullet, NULL, &hrac->naboj);
    }
}
