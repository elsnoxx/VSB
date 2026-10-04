#include "hra.h"
#include "constants.h"

void pohyb_botu(hrac hrac1, hrac hrac2, menu game_menu, hrac boty[], double delta_time, SDL_Rect domek)
{
    SDL_Rect novaPoloha;
    switch (game_menu.pocet_hracu)
    {
    case 1:

        for (int i = 0; i < 3; ++i)
        {
            if (SDL_HasIntersection(&boty[i].poloha, &domek))
            {
                if (boty[i].smer == 'r')
                {
                    if (boty[i].poloha.y > domek.y / 2)
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                }
                else if (boty[i].smer == 'l')
                {
                    if (boty[i].poloha.y > domek.y / 2)
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.y += 10 * delta_time;
                    }
                }
                else if (boty[i].smer == 'u')
                {
                    if (boty[i].poloha.x > domek.y / 2)
                    {
                        boty[i].poloha.x -= 25 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.x -= 25 * delta_time;
                    }
                }
                else if (boty[i].smer == 'd')
                {
                    if (boty[i].poloha.x > domek.y / 2)
                    {
                        boty[i].poloha.x += 75 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.x -= 75 * delta_time;
                    }
                }
            }
            else
            {
                if (hrac1.poloha.x > boty[i].poloha.x)
                {
                    boty[i].poloha.x += 75 * delta_time;
                    boty[i].smer = 'r';
                }
                else if (hrac1.poloha.x < boty[i].poloha.x)
                {
                    boty[i].poloha.x -= 25 * delta_time;
                    boty[i].smer = 'l';
                }

                if (hrac1.poloha.y > boty[i].poloha.y)
                {
                    boty[i].poloha.y += 75 * delta_time;
                    boty[i].smer = 'd';
                }
                else if (hrac1.poloha.y < boty[i].poloha.y)
                {
                    boty[i].poloha.y -= 10 * delta_time;
                    boty[i].smer = 'u';
                }
            }
        }
        break;
    case 2:
        for (int i = 0; i < 2; i++)
        {
            if (SDL_HasIntersection(&boty[i].poloha, &domek))
            {
                if (boty[i].smer == 'r')
                {
                    if (boty[i].poloha.y > domek.y / 2)
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                }
                else if (boty[i].smer == 'l')
                {
                    if (boty[i].poloha.y > domek.y / 2)
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.y -= 10 * delta_time;
                    }
                }
                else if (boty[i].smer == 'u')
                {
                    if (boty[i].poloha.x > domek.y / 2)
                    {
                        boty[i].poloha.x -= 25 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.x -= 25 * delta_time;
                    }
                }
                else if (boty[i].smer == 'd')
                {
                    if (boty[i].poloha.x > domek.y / 2)
                    {
                        boty[i].poloha.x += 75 * delta_time;
                    }
                    else
                    {
                        boty[i].poloha.x -= 75 * delta_time;
                    }
                }
            }
        }

        if (hrac1.poloha.x > boty[0].poloha.x)
        {
            boty[0].poloha.x += 75 * delta_time;
            boty[0].smer = 'r';
        }
        else if (hrac1.poloha.x < boty[0].poloha.x)
        {
            boty[0].poloha.x -= 25 * delta_time;
            boty[0].smer = 'l';
        }

        if (hrac1.poloha.y > boty[0].poloha.y)
        {
            boty[0].poloha.y += 75 * delta_time;
            boty[0].smer = 'd';
        }
        else if (hrac1.poloha.y < boty[0].poloha.y)
        {
            boty[0].poloha.y -= 10 * delta_time;
            boty[0].smer = 'u';
        }

        if (hrac2.poloha.x > boty[1].poloha.x)
        {
            boty[1].poloha.x += 75 * delta_time;
            boty[1].smer = 'r';
        }
        else if (hrac2.poloha.x < boty[1].poloha.x)
        {
            boty[1].poloha.x -= 25 * delta_time;
            boty[1].smer = 'l';
        }

        if (hrac2.poloha.y > boty[1].poloha.y)
        {
            boty[1].poloha.y += 75 * delta_time;
            boty[1].smer = 'd';
        }
        else if (hrac2.poloha.y < boty[1].poloha.y)
        {
            boty[1].poloha.y -= 10 * delta_time;
            boty[1].smer = 'u';
        }

        break;

    default:
        break;
    }
}
