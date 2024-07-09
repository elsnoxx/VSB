#ifndef HRA_H
#define HRA_H
#include <SDL2/SDL.h>
#include <stdbool.h>
#include <time.h>
#include <SDL2/SDL_image.h>
#include <SDL2/SDL_ttf.h>


typedef struct
{
    int pocet_hracu;
    int minut;
} menu;

typedef struct
{
    int score;
    char smer;
    bool zasah;
    char smer_strely;
    SDL_Rect naboj;
    bool vystrel;
    SDL_Rect poloha;
} hrac;

void sdl_draw_text(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color, SDL_Rect location, const char *text);
SDL_Rect random_position(SDL_Rect domcek);
void uklizeni(SDL_Renderer *renderer,SDL_Window *window,SDL_Texture *bullet, SDL_Texture *house,SDL_Texture *bush, SDL_Texture *rock,SDL_Texture *background,SDL_Texture *red_down,SDL_Texture *red_left,SDL_Texture *red_right,SDL_Texture *red_up,SDL_Texture *pink_down,SDL_Texture *pink_left,SDL_Texture *pink_right,SDL_Texture *pink_up,SDL_Texture *gray_down,SDL_Texture *gray_left,SDL_Texture *gray_right,SDL_Texture *gray_up);
void event_handeling(SDL_Event event, int *running, int *mouseX_click, int *mouseY_click, int screen, double delta_time, hrac *hrac1, hrac *hrac2, menu game_menu, SDL_Rect domek);
void strelba(hrac *hrac);
void pohyb_botu(hrac hrac1, hrac hrac2, menu game_menu, hrac boty[], double delta_time, SDL_Rect domek);


// funkce pro renderovani obrazku
void renderPlayer(SDL_Renderer *renderer, hrac hrac, SDL_Texture *textures[]);
void render_strelby(SDL_Renderer *renderer, hrac *hrac, SDL_Texture *bullet, SDL_Rect domek, double delta_time);


// menu hry
void menu_screen(hrac boty[],SDL_Rect domek,int *update,SDL_Renderer *renderer, TTF_Font *font, SDL_Color color_text, SDL_Color color_nadpis, SDL_Color color_chose,int *screen, Uint32 *startTime, menu *game_menu, int mouseX_click, int mouseY_click);
void render_menu(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color_text, SDL_Color color_nadpis, SDL_Color color_chose, menu game_menu, SDL_Rect nadpis, SDL_Rect jeden_hrac, SDL_Rect dva_hraci, SDL_Rect minuta, SDL_Rect dve_minuta, SDL_Rect tri_minuta, SDL_Rect hra);
void logick_menu(int *screen, Uint32 *startTime, menu *game_menu, int mouseX_click, int mouseY_click, SDL_Rect jeden_hrac, SDL_Rect dva_hraci, SDL_Rect minuta, SDL_Rect dve_minuta, SDL_Rect tri_minuta, SDL_Rect hra);
void init_bots(menu game_menu,SDL_Rect domek,hrac boty[]);

// end hry
void end_screen(int *mouseX_click, int *mouseY_click, int *screen, int *update,hrac hrac1,hrac hrac2,hrac *boty,SDL_Renderer *renderer, menu game_menu,TTF_Font *font,SDL_Color color_nadpis, SDL_Color color_chose,SDL_Color color_text);
void saveScore(hrac hraci[], menu game_menu);
void vypis_score(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color_text, SDL_Color color_nadpis);
void vypis_score_hraci(SDL_Renderer *renderer, TTF_Font *font, hrac hraci[], SDL_Color color_text, char *hraci_nazvy[], SDL_Color color_nadpis, SDL_Color color_chose);

#endif