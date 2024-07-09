#include "hra.h"
#include "constants.h"

void sdl_draw_text(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color, SDL_Rect location, const char *text)
{
    // Vykreslení textu se zadaným fontem a barvou do obrázku (surface)
    SDL_Surface *surface = TTF_RenderText_Blended(font, text, color);
    // Převod surface na hardwarovou texturu
    SDL_Texture *texture = SDL_CreateTextureFromSurface(renderer, surface);

    // Vykreslení obrázku
    SDL_RenderCopy(renderer, texture, NULL, &location);

    // Uvolnění textury a surface
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}
SDL_Rect random_position(SDL_Rect domek)
{
    SDL_Rect nova_pozice;

    nova_pozice.x = rand() % (WINDOW_WIDHT - PILLOW_WIDHT);
    nova_pozice.y = rand() % (WINDOW_HEIGHT - PILLOW_HEIGHT);

    while (SDL_HasIntersection(&domek, &nova_pozice))
    {
        nova_pozice.x = rand() % (WINDOW_WIDHT - PILLOW_WIDHT);
        nova_pozice.y = rand() % (WINDOW_HEIGHT - PILLOW_HEIGHT);
    }

    nova_pozice.w = PILLOW_WIDHT;
    nova_pozice.h = PILLOW_HEIGHT;
    return nova_pozice;
}


void strelba(hrac *hrac)
{
    if (!hrac->vystrel)
    {
        // Nastavte stav střelby na true
        hrac->vystrel = true;

        // Inicializace střely
        hrac->naboj.x = hrac->poloha.x + hrac->poloha.w / 2;
        hrac->naboj.y = hrac->poloha.y + hrac->poloha.h / 2;
        hrac->naboj.w = 8;
        hrac->naboj.h = 15;
        hrac->smer_strely = hrac->smer;
        switch (hrac->smer)
        {
        case 'r':
            hrac->naboj.x += 30;
            break;
        case 'l':
            hrac->naboj.x -= 30;
            break;
        case 'd':
            hrac->naboj.y += 30;
            break;
        case 'u':
            hrac->naboj.y -= 30;
            break;
        default:
            break;
        }
    }
}

void event_handeling(SDL_Event event, int *running, int *mouseX_click, int *mouseY_click, int screen, double delta_time, hrac *hrac1, hrac *hrac2, menu game_menu, SDL_Rect domek)
{

    while (SDL_PollEvent(&event))
    {
        // mouseX_click = 0;
        // mouseY_click = 0;
        // Pokud došlo k uzavření okna, nastav proměnnou `running` na `0`
        if (event.type == SDL_QUIT)
        {
            *running = 0;
        }
        else if (event.type == SDL_MOUSEMOTION)
        {
            // Zpracování pohybu myši
            int mouseX = event.motion.x;
            int mouseY = event.motion.y;
        }
        else if (event.type == SDL_MOUSEBUTTONDOWN)
        {
            // Zpracování stisknutí tlačítka myši
            if (event.button.button == SDL_BUTTON_LEFT)
            {
                *mouseX_click = event.motion.x;
                *mouseY_click = event.motion.y;
            }
        }
        else if (event.type == SDL_KEYDOWN && screen == 1)
        {
            switch (event.key.keysym.sym)
            {
            case SDLK_UP:
                if (hrac1->poloha.y - 200 * delta_time >= 0 && !SDL_HasIntersection(&hrac1->poloha, &domek))
                {
                    SDL_Rect novaPoloha = {hrac1->poloha.x, hrac1->poloha.y - 200 * delta_time, hrac1->poloha.w, hrac1->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac1->poloha.y = novaPoloha.y;
                        hrac1->smer = 'u';
                    }
                }
                break;
            case SDLK_DOWN:
                if (hrac1->poloha.y + 200 * delta_time <= WINDOW_HEIGHT && !SDL_HasIntersection(&hrac1->poloha, &domek))
                {
                    SDL_Rect novaPoloha = {hrac1->poloha.x, hrac1->poloha.y + 200 * delta_time, hrac1->poloha.w, hrac1->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac1->poloha.y = novaPoloha.y;
                        hrac1->smer = 'd';
                    }
                }
                break;
            case SDLK_LEFT:
                if (hrac1->poloha.x - 200 * delta_time >= 0 && !SDL_HasIntersection(&hrac1->poloha, &domek))
                {
                    SDL_Rect novaPoloha = {hrac1->poloha.x - 200 * delta_time, hrac1->poloha.y, hrac1->poloha.w, hrac1->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac1->poloha.x = novaPoloha.x;
                        hrac1->smer = 'l';
                    }
                }
                break;
            case SDLK_RIGHT:
                if (hrac1->poloha.x + 200 * delta_time <= WINDOW_WIDHT && !SDL_HasIntersection(&hrac1->poloha, &domek))
                {
                    SDL_Rect novaPoloha = {hrac1->poloha.x + 200 * delta_time, hrac1->poloha.y, hrac1->poloha.w, hrac1->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac1->poloha.x = novaPoloha.x;
                        hrac1->smer = 'r';
                    }
                }
                break;
            case SDLK_d:
                if (hrac2->poloha.x + 200 * delta_time <= WINDOW_WIDHT && game_menu.pocet_hracu == 2)
                {
                    SDL_Rect novaPoloha = {hrac2->poloha.x + 200 * delta_time, hrac2->poloha.y, hrac2->poloha.w, hrac2->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac2->poloha.x += 200 * delta_time;
                        hrac2->smer = 'r';
                    }
                }
                break;
            case SDLK_a:
                if (hrac2->poloha.x - 200 * delta_time >= 0 && game_menu.pocet_hracu == 2)
                {
                    SDL_Rect novaPoloha = {hrac2->poloha.x - 200 * delta_time, hrac2->poloha.y, hrac2->poloha.w, hrac2->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac2->poloha.x -= 200 * delta_time;
                        hrac2->smer = 'l';
                    }
                }
                break;
            case SDLK_s:
                if (hrac2->poloha.y + 200 * delta_time <= WINDOW_HEIGHT && game_menu.pocet_hracu == 2)
                {
                    SDL_Rect novaPoloha = {hrac2->poloha.x, hrac2->poloha.y + 200 * delta_time, hrac2->poloha.w, hrac2->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac2->poloha.y += 200 * delta_time;
                        hrac2->smer = 'd';
                    }
                }
                break;
            case SDLK_w:
                if (hrac2->poloha.y - 200 * delta_time >= 0 && game_menu.pocet_hracu == 2)
                {
                    SDL_Rect novaPoloha = {hrac2->poloha.x, hrac2->poloha.y - 200 * delta_time, hrac2->poloha.w, hrac2->poloha.h};
                    if (!SDL_HasIntersection(&domek, &novaPoloha))
                    {
                        hrac2->poloha.y -= 200 * delta_time;
                        hrac2->smer = 'u';
                    }
                }
                break;
            case SDLK_m:
                strelba(hrac1);
                break;
            case SDLK_r:
                strelba(hrac2);
                break;
            }
            // printf("hrac1 je na x %d a y %d\nhrac2 je na x %d a y %d\n", hrac1.poloha.x, hrac1.poloha.y, hrac2.poloha.x, hrac2.poloha.y);
        }
    }
}

void uklizeni(SDL_Renderer *renderer, SDL_Window *window, SDL_Texture *bullet, SDL_Texture *house, SDL_Texture *bush, SDL_Texture *rock, SDL_Texture *background, SDL_Texture *red_down, SDL_Texture *red_left, SDL_Texture *red_right, SDL_Texture *red_up, SDL_Texture *pink_down, SDL_Texture *pink_left, SDL_Texture *pink_right, SDL_Texture *pink_up, SDL_Texture *gray_down, SDL_Texture *gray_left, SDL_Texture *gray_right, SDL_Texture *gray_up)
{
    SDL_DestroyTexture(house);
    SDL_DestroyTexture(background);
    SDL_DestroyTexture(bush);
    SDL_DestroyTexture(rock);

    SDL_DestroyTexture(red_down);
    SDL_DestroyTexture(red_left);
    SDL_DestroyTexture(red_right);
    SDL_DestroyTexture(red_up);
    SDL_DestroyTexture(pink_down);
    SDL_DestroyTexture(pink_left);
    SDL_DestroyTexture(pink_right);
    SDL_DestroyTexture(pink_up);
    SDL_DestroyTexture(gray_down);
    SDL_DestroyTexture(gray_left);
    SDL_DestroyTexture(gray_right);
    SDL_DestroyTexture(gray_up);
    SDL_DestroyTexture(bullet);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}