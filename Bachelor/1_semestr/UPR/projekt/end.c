#include "hra.h"

// hlavni funkce pro zobrazeni konecne plochy
void end_screen(int *mouseX_click, int *mouseY_click, int *screen, int *update,hrac hrac1,hrac hrac2,hrac *boty,SDL_Renderer *renderer, menu game_menu,TTF_Font *font,SDL_Color color_nadpis, SDL_Color color_chose,SDL_Color color_text)
{
    SDL_RenderClear(renderer);
    SDL_Rect end = {
        .x = 400,
        .y = 10,
        .w = 400,
        .h = 200};

    SDL_Rect again = {
        .x = 450,
        .y = 670,
        .w = 300,
        .h = 100};

    if ((*mouseX_click >= again.x && *mouseX_click <= again.x + again.w) &&
        (*mouseY_click >= again.y && *mouseY_click <= again.y + again.h))
    {
        *screen = 0;
        *mouseX_click = 0;
        *mouseY_click = 0;
        // printf("%d\n", screen);
    }
    // printf("%d\n",*update);
    sdl_draw_text(renderer, font, color_nadpis, end, "Konec");
    sdl_draw_text(renderer, font, color_text, again, "Znova");
    if (game_menu.pocet_hracu == 1)
    {
        hrac hraci[] = {hrac1, boty[0], boty[1], boty[2]};
        char *hraci_nazvy[] = {"Hrac 1", "Bot 1", "Bot 2", "Bot 3"};
        if (*update == 0)
        {
            saveScore(hraci, game_menu);
            *update = 1;
        }
        vypis_score_hraci(renderer, font, hraci, color_text, hraci_nazvy, color_nadpis, color_chose);
    }
    else if (game_menu.pocet_hracu == 2)
    {
        hrac hraci[] = {hrac1, hrac2, boty[0], boty[1]};
        char *hraci_nazvy[] = {"Hrac 1", "Hrac 2", "Bot 2", "Bot 3"};
        if (*update == 0)
        {
            saveScore(hraci, game_menu);
            *update = 1;
        }
        vypis_score_hraci(renderer, font, hraci, color_text, hraci_nazvy, color_nadpis, color_chose);
    }
    vypis_score(renderer, font, color_text, color_nadpis);
}

// ulozeni skore do souboru
void saveScore(hrac hraci[], menu game_menu)
{
    int scores[5];
    FILE *file = fopen("score.txt", "r");
    for (int i = 0; i < 5; ++i)
    {
        fscanf(file, "%d", &scores[i]);
    }
    fclose(file);
    int noupdate = 0;
    int index;
    for (int i = 0; i < game_menu.pocet_hracu; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            if (hraci[i].score == scores[j])
            {
                noupdate = 1;
            }

            if (hraci[i].score > scores[j])
            {
                index = j;
                break; // Nalezena pozice pro nové skóre
            }
        }
        if (noupdate == 0)
        {
            for (int k = 4; k > index; k--)
            {
                scores[k] = scores[k - 1];
            }
            scores[index] = hraci[i].score;
        }
        noupdate = 0;
    }

    file = fopen("score.txt", "w");
    for (int i = 0; i < 5; i++)
    {
        fprintf(file, "%d\n", scores[i]);
    }
    fclose(file);
}
// vypis skore do souboru
void vypis_score(SDL_Renderer *renderer, TTF_Font *font, SDL_Color color_text, SDL_Color color_nadpis)
{
    FILE *file = fopen("score.txt", "r");
    int scores[5];
    for (int i = 0; i < 5; ++i)
    {
        fscanf(file, "%d", &scores[i]);
    }
    fclose(file);
    SDL_Rect nadpis = {.x = 50, .y = 300, .w = 400, .h = 100};
    sdl_draw_text(renderer, font, color_nadpis, nadpis, "Nejlepsi skore");
    // Vypisujeme pouze 5 nejvyšších skóre
    for (int i = 1; i <= 5; ++i)
    {
        char text[20];
        snprintf(text, sizeof(text), "Score %d: %d", i, scores[i - 1]);

        SDL_Rect textLocation = {.x = 100, .y = 386 + i * 50, .w = 200, .h = 50};
        sdl_draw_text(renderer, font, color_text, textLocation, text);
    }
}
// najde a vrati nejlepsi skore
int best_score(hrac hraci[])
{
    int best = 0;
    for (int i = 0; i < 4; i++)
    {
        if (hraci[i].score > best)
        {
            best = hraci[i].score;
        }
    }
    return best;
}
// vypis skore na plochu
void vypis_score_hraci(SDL_Renderer *renderer, TTF_Font *font, hrac hraci[], SDL_Color color_text, char *hraci_nazvy[], SDL_Color color_nadpis, SDL_Color color_chose)
{
    int best = best_score(hraci);
    SDL_Rect nadpis = {.x = 700, .y = 300, .w = 400, .h = 100};
    sdl_draw_text(renderer, font, color_nadpis, nadpis, "Hreni skore");
    for (int i = 0; i < 4; ++i)
    {
        char text[50];
        snprintf(text, sizeof(text), "%s: %d", hraci_nazvy[i], hraci[i].score);
        SDL_Rect textLocation = {.x = 750, .y = 400 + i * 50, .w = 200, .h = 50};
        if (hraci[i].score == best)
        {
            sdl_draw_text(renderer, font, color_chose, textLocation, text);
        }
        else
        {
            sdl_draw_text(renderer, font, color_text, textLocation, text);
        }
    }
}