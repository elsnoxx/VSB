#include "hra.h"
#include "constants.h"

int main()
{
    // Inicializace SDL
    SDL_Init(SDL_INIT_VIDEO);
    TTF_Init();
    

    // Vytvoření okna
    SDL_Window *window = SDL_CreateWindow(
        "Bulanci",       // Titulek okna
        0,               // Souřadnice x
        0,               // Souřadnice y
        WINDOW_WIDHT,    // Šířka
        WINDOW_HEIGHT,   // Výška
        SDL_WINDOW_SHOWN // Okno se má po vytvoření rovnou zobrazit
    );

    // Vytvoření kreslítka
    SDL_Renderer *renderer = SDL_CreateRenderer(
        window,
        -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    // nacteni obrazku na pozadi a nacteni obrazku postav
    SDL_Texture *background = IMG_LoadTexture(renderer, "plocha.jpg");
    SDL_Texture *house = IMG_LoadTexture(renderer, "domek.png");
    SDL_Texture *rock = IMG_LoadTexture(renderer, "kamen.png");
    SDL_Texture *bush = IMG_LoadTexture(renderer, "ker.png");

    SDL_Texture *red_down = IMG_LoadTexture(renderer, "red_down.png");
    SDL_Texture *red_up = IMG_LoadTexture(renderer, "red_up.png");
    SDL_Texture *red_left = IMG_LoadTexture(renderer, "red_left.png");
    SDL_Texture *red_right = IMG_LoadTexture(renderer, "red_right.png");

    SDL_Texture *pink_down = IMG_LoadTexture(renderer, "pink_down.png");
    SDL_Texture *pink_up = IMG_LoadTexture(renderer, "pink_up.png");
    SDL_Texture *pink_left = IMG_LoadTexture(renderer, "pink_left.png");
    SDL_Texture *pink_right = IMG_LoadTexture(renderer, "pink_right.png");

    SDL_Texture *gray_down = IMG_LoadTexture(renderer, "gray_down.png");
    SDL_Texture *gray_up = IMG_LoadTexture(renderer, "gray_up.png");
    SDL_Texture *gray_left = IMG_LoadTexture(renderer, "gray_left.png");
    SDL_Texture *gray_right = IMG_LoadTexture(renderer, "gray_right.png");
    SDL_Texture *bullet = IMG_LoadTexture(renderer, "bullet.png");

    // pro hru
    SDL_Rect domek = {
        .x = 300,
        .y = 300,
        .w = 338,
        .h = 326};

    SDL_Event event;
    hrac boty[3];

    Uint64 now = SDL_GetPerformanceCounter();
    Uint64 last = 0;

    // promenne zarucujici ovladani a chod hry
    double delta_time = 0;
    int running = 1;
    int update = 0;
    float line_x = 100;
    int screen = 0;
    int mouseX_click = 0;
    int mouseY_click = 0;

    // nastaveni herniho menu
    menu game_menu;
    game_menu.pocet_hracu = 1;
    game_menu.minut = 1;

    hrac hrac1;
    hrac1.smer = 'r';
    hrac1.score = 0;
    hrac1.poloha = random_position(domek);
    hrac1.poloha.h = PILLOW_HEIGHT;
    hrac1.poloha.w = PILLOW_WIDHT;
    hrac1.vystrel = false;

    hrac hrac2;
    hrac2.smer = 'r';
    hrac2.score = 0;
    hrac2.poloha = random_position(domek);
    hrac2.poloha.h = PILLOW_HEIGHT;
    hrac2.poloha.w = PILLOW_WIDHT;
    hrac1.vystrel = false;

    SDL_Texture *player1Textures[] = {red_right, red_left, red_down, red_up};
    SDL_Texture *player2Textures[] = {pink_right, pink_left, pink_down, pink_up};
    SDL_Texture *botTextures[] = {gray_right, gray_left, gray_down, gray_up};

    // hlavni zpusoby vypisu barev
    TTF_Font *font = TTF_OpenFont("Arial.ttf", 20);
    SDL_Color color_nadpis = {255, 0, 0, 255};
    SDL_Color color_text = {255, 255, 255, 255};
    SDL_Color color_chose = {0, 0, 255, 255};
    // printf("%d, %d, %d, %d",hrac1.poloha.h,hrac1.poloha.w,hrac1.poloha.x,hrac1.poloha.y);
    // pouzito pro casove ukonceni hry
    Uint32 startTime = 0;
    Uint32 gameTimeLimit = 0;
    Uint32 currentTime = 0;
    Uint32 elapsedTime = 0;

    // pouzito pro casove ukonceni hry
    while (running == 1)
    {
        // vypocet delta time pro zlepseni posunu objektu a synchronizaci s obrazovkou
        last = now;
        now = SDL_GetPerformanceCounter();
        delta_time = (double)((now - last) / (double)SDL_GetPerformanceFrequency());
        // funkce, ktera se stara o veskere udalosti
        event_handeling(event, &running, &mouseX_click, &mouseY_click, screen, delta_time, &hrac1, &hrac2, game_menu, domek);

        // Nastav barvu vykreslování na černou
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);

        switch (screen)
        {
        case 0:
            // plocha menu
            menu_screen(boty, domek, &update, renderer, font, color_text, color_nadpis, color_chose, &screen, &startTime, &game_menu, mouseX_click, mouseY_click);
            break;
        case 1:
            //  plocha hry
            // uklizeni pozadí
            SDL_RenderClear(renderer);

            gameTimeLimit = game_menu.minut * 60000;

            pohyb_botu(hrac1, hrac2, game_menu, boty, delta_time, domek);

            // render_hry(renderer, game_menu, hrac1, hrac2, boty, domek, bullet, house, bush, rock, background, delta_time, player1Textures, player2Textures, botTextures);
            SDL_Rect ker = {
                .x = 900,
                .y = 170,
                .w = 131,
                .h = 113};

            SDL_Rect kamen = {
                .x = 100,
                .y = 150,
                .w = 99,
                .h = 92};

            SDL_Rect plocha = {
                .x = 0,
                .y = 0,
                .w = WINDOW_WIDHT,
                .h = WINDOW_HEIGHT};
            SDL_RenderCopy(renderer, background, NULL, &plocha);
            SDL_RenderCopy(renderer, house, NULL, &domek);
            SDL_RenderCopy(renderer, bush, NULL, &ker);
            SDL_RenderCopy(renderer, rock, NULL, &kamen);

            if (game_menu.pocet_hracu == 1)
            {
                for (int i = 0; i < 3; i++)
                {
                    if (hrac1.vystrel && SDL_HasIntersection(&hrac1.naboj, &boty[i].poloha))
                    {
                        boty[i].zasah = true;
                        hrac1.score += 10;
                        boty[i].poloha = random_position(domek);
                        hrac1.vystrel = false;
                    }
                    if (boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &hrac1.poloha))
                    {
                        hrac1.zasah = true;
                        boty[i].score += 10;
                        hrac1.poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }
                    if (i != 0 && boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &boty[0].poloha))
                    {
                        boty[0].zasah = true;
                        boty[i].score += 10;
                        boty[0].poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }
                    if (i != 1 && boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &boty[1].poloha))
                    {
                        boty[1].zasah = true;
                        boty[i].score += 10;
                        boty[1].poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }
                    if (i != 2 && boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &boty[2].poloha))
                    {
                        boty[2].zasah = true;
                        boty[i].score += 10;
                        boty[2].poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }

                    if (boty[i].poloha.x == hrac1.poloha.x || boty[i].poloha.y == hrac1.poloha.y 
                    || boty[1].poloha.y == boty[0].poloha.y || boty[1].poloha.x == boty[0].poloha.x
                    || boty[2].poloha.y == boty[0].poloha.y || boty[2].poloha.x == boty[0].poloha.x)
                    {
                        strelba(&boty[i]);
                    }
                    renderPlayer(renderer, boty[i], botTextures);
                    render_strelby(renderer, &boty[i], bullet, domek, delta_time);
                }
                renderPlayer(renderer, hrac1, player1Textures);

                render_strelby(renderer, &hrac1, bullet, domek, delta_time);
            }
            else if (game_menu.pocet_hracu == 2)
            {
                if (hrac1.vystrel && SDL_HasIntersection(&hrac1.naboj, &hrac2.poloha))
                {
                    hrac1.zasah = true;
                    hrac1.score += 10;
                    hrac2.poloha = random_position(domek);
                    hrac1.vystrel = false;
                }
                else if (hrac2.vystrel && SDL_HasIntersection(&hrac2.naboj, &hrac1.poloha))
                {
                    hrac2.zasah = true;
                    hrac2.score += 10;
                    hrac1.poloha = random_position(domek);
                    hrac2.vystrel = false;
                }

                for (int i = 0; i < 2; i++)
                {
                    if (hrac1.vystrel && SDL_HasIntersection(&hrac1.naboj, &boty[i].poloha))
                    {
                        boty[i].zasah = true;
                        hrac1.score += 10;
                        boty[i].poloha = random_position(domek);
                        hrac1.vystrel = false;
                    }
                    if (boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &hrac1.poloha))
                    {
                        hrac1.zasah = true;
                        boty[i].score += 10;
                        hrac1.poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }
                    if (hrac2.vystrel && SDL_HasIntersection(&hrac2.naboj, &boty[i].poloha))
                    {
                        boty[i].zasah = true;
                        hrac2.score += 10;
                        boty[i].poloha = random_position(domek);
                        hrac2.vystrel = false;
                    }
                    if (boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &hrac2.poloha))
                    {
                        hrac2.zasah = true;
                        boty[i].score += 10;
                        hrac2.poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }
                    if (i != 0 && boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &boty[0].poloha))
                    {
                        boty[0].zasah = true;
                        boty[i].score += 10;
                        boty[0].poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }
                    if (i != 1 && boty[i].vystrel && SDL_HasIntersection(&boty[i].naboj, &boty[1].poloha))
                    {
                        boty[1].zasah = true;
                        boty[i].score += 10;
                        boty[1].poloha = random_position(domek);
                        boty[i].vystrel = false;
                    }

                    if (boty[i].poloha.x == hrac1.poloha.x || boty[i].poloha.y == hrac1.poloha.y || boty[1].poloha.y == boty[0].poloha.y || boty[1].poloha.x == boty[0].poloha.x)
                    {
                        strelba(&boty[i]);
                    }
                    render_strelby(renderer, &boty[i], bullet, domek, delta_time);
                    renderPlayer(renderer, boty[i], botTextures);
                    render_strelby(renderer, &boty[i], bullet, domek, delta_time);
                }
                renderPlayer(renderer, hrac1, player1Textures);
                renderPlayer(renderer, hrac2, player2Textures);
                render_strelby(renderer, &hrac1, bullet, domek, delta_time);
                render_strelby(renderer, &hrac2, bullet, domek, delta_time);
            }
            currentTime = SDL_GetTicks();
            elapsedTime = currentTime - startTime;
            // printf("gameTimeLimit %d, startTime %d, currentTime %d, elapsedTime %d\n",gameTimeLimit,startTime,currentTime,elapsedTime);
            if (elapsedTime >= gameTimeLimit)
            {
                // printf("Konec hry!\n");
                screen = 2;
            }
            break;
        case 2:
            // plocha vysledku
            // printf("%d\n",update);
            end_screen(&mouseX_click, &mouseY_click, &screen, &update, hrac1, hrac2, boty, renderer, game_menu, font, color_nadpis, color_chose, color_text);
            break;
        default:
            break;
        }
        // Zobraz vykreslené prvky na obrazovku
        SDL_RenderPresent(renderer);
    }

    // Uvolnění prostředků
    uklizeni(renderer, window, bullet, house, bush, rock, background, red_down, red_left, red_right, red_up, pink_down, pink_left, pink_right, pink_up, gray_down, gray_left, gray_right, gray_up);

    return 0;
}