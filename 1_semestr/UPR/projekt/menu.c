#include "hra.h"
#include "constants.h"

void menu_screen(hrac boty[],SDL_Rect domek,int *update,SDL_Renderer *renderer, TTF_Font *font, SDL_Color color_text, SDL_Color color_nadpis, SDL_Color color_chose,int *screen, Uint32 *startTime, menu *game_menu, int mouseX_click, int mouseY_click)
{
    SDL_RenderClear(renderer);
    *update = 0;
    SDL_Rect nadpis = {
        .x = 400,
        .y = 10,
        .w = 400,
        .h = 200};
    SDL_Rect jeden_hrac = {
        .x = 450,
        .y = 250,
        .w = 300,
        .h = 100};
    SDL_Rect dva_hraci = {
        .x = 450,
        .y = 350,
        .w = 300,
        .h = 100};
    SDL_Rect minuta = {
        .x = 150,
        .y = 600,
        .w = 200,
        .h = 100};
    SDL_Rect dve_minuta = {
        .x = 400,
        .y = 600,
        .w = 300,
        .h = 100};
    SDL_Rect tri_minuta = {
        .x = 750,
        .y = 600,
        .w = 300,
        .h = 100};
    SDL_Rect hra = {
        .x = 450,
        .y = 450,
        .w = 300,
        .h = 100};
    // logicky vyber nastaveni hry
    logick_menu(screen, startTime, game_menu, mouseX_click, mouseY_click, jeden_hrac, dva_hraci, minuta, dve_minuta, tri_minuta, hra);
    // render plochy menu
    render_menu(renderer, font, color_text, color_nadpis, color_chose, *game_menu, nadpis, jeden_hrac, dva_hraci, minuta, dve_minuta, tri_minuta, hra);
    // inicializace botu
    init_bots(*game_menu, domek, boty);
}

// funkce pro renderovani menu hry
void render_menu(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color_text, SDL_Color color_nadpis, SDL_Color color_chose, menu game_menu, SDL_Rect nadpis, SDL_Rect jeden_hrac, SDL_Rect dva_hraci, SDL_Rect minuta, SDL_Rect dve_minuta, SDL_Rect tri_minuta, SDL_Rect hra)
{
    sdl_draw_text(renderer, font, color_nadpis, nadpis, "Bulanci");
    if (game_menu.pocet_hracu == 1)
    {
        sdl_draw_text(renderer, font, color_chose, jeden_hrac, "Jeden hrac");
        sdl_draw_text(renderer, font, color_text, dva_hraci, "Dva hraci");
    }
    else if (game_menu.pocet_hracu == 2)
    {
        sdl_draw_text(renderer, font, color_text, jeden_hrac, "Jeden hrac");
        sdl_draw_text(renderer, font, color_chose, dva_hraci, "Dva hraci");
    }

    if (game_menu.minut == 1)
    {
        sdl_draw_text(renderer, font, color_chose, minuta, "Minuta");
        sdl_draw_text(renderer, font, color_text, dve_minuta, "Dve minuty");
        sdl_draw_text(renderer, font, color_text, tri_minuta, "Tri minuty");
    }
    else if (game_menu.minut == 2)
    {
        sdl_draw_text(renderer, font, color_text, minuta, "Minuta");
        sdl_draw_text(renderer, font, color_chose, dve_minuta, "Dve minuty");
        sdl_draw_text(renderer, font, color_text, tri_minuta, "Tri minuty");
    }
    else if (game_menu.minut == 3)
    {
        sdl_draw_text(renderer, font, color_text, minuta, "Minuta");
        sdl_draw_text(renderer, font, color_text, dve_minuta, "Dve minuty");
        sdl_draw_text(renderer, font, color_chose, tri_minuta, "Tri minuty");
    }
    sdl_draw_text(renderer, font, color_nadpis, hra, "Hraj !");
}
// funkce, ktera resi volby hernich rezimu
void logick_menu(int *screen, Uint32 *startTime, menu *game_menu, int mouseX_click, int mouseY_click, SDL_Rect jeden_hrac, SDL_Rect dva_hraci, SDL_Rect minuta, SDL_Rect dve_minuta, SDL_Rect tri_minuta, SDL_Rect hra)
{
    if ((mouseX_click >= tri_minuta.x && mouseX_click <= tri_minuta.x + tri_minuta.w) &&
        (mouseY_click >= tri_minuta.y && mouseY_click <= tri_minuta.y + tri_minuta.h))
    {
        game_menu->minut = 3;
    }
    else if ((mouseX_click >= dve_minuta.x && mouseX_click <= dve_minuta.x + dve_minuta.w) &&
             (mouseY_click >= dve_minuta.y && mouseY_click <= dve_minuta.y + dve_minuta.h))
    {
        game_menu->minut = 2;
    }
    else if ((mouseX_click >= minuta.x && mouseX_click <= minuta.x + minuta.w) &&
             (mouseY_click >= minuta.y && mouseY_click <= minuta.y + minuta.h))
    {
        game_menu->minut = 1;
    }

    if ((mouseX_click >= jeden_hrac.x && mouseX_click <= jeden_hrac.x + jeden_hrac.w) &&
        (mouseY_click >= jeden_hrac.y && mouseY_click <= jeden_hrac.y + jeden_hrac.h))
    {
        game_menu->pocet_hracu = 1;
    }
    else if ((mouseX_click >= dva_hraci.x && mouseX_click <= dva_hraci.x + dva_hraci.w) &&
             (mouseY_click >= dva_hraci.y && mouseY_click <= dva_hraci.y + dva_hraci.h))
    {
        game_menu->pocet_hracu = 2;
    }

    if ((mouseX_click >= hra.x && mouseX_click <= hra.x + hra.w) &&
        (mouseY_click >= hra.y && mouseY_click <= hra.y + hra.h))
    {
        *screen = 1;
        mouseX_click = 0;
        mouseY_click = 0;
        *startTime = SDL_GetTicks();
    }
}
//funkce, ktera inicializuje boty podle herniho menu
void init_bots(menu game_menu,SDL_Rect domek,hrac boty[])
{
    if (game_menu.pocet_hracu == 1)
    {
        for (int i = 0; i < 3; ++i)
        {
            boty[i].smer = 'r';
            boty[i].score = 0;
            boty[i].poloha = random_position(domek);
            boty[i].poloha.h = PILLOW_HEIGHT;
            boty[i].poloha.w = PILLOW_WIDHT;
            boty[i].vystrel = false;
        }
    }
    else if (game_menu.pocet_hracu == 2)
    {
        for (int i = 0; i < 2; ++i)
        {
            boty[i].smer = 'r';
            boty[i].score = 0;
            boty[i].poloha = random_position(domek);
            boty[i].poloha.h = PILLOW_HEIGHT;
            boty[i].poloha.w = PILLOW_WIDHT;
            boty[i].vystrel = false;
        }
    }
}