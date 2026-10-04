#include <stdio.h>
#include <stdbool.h>
#include "drawing.h"

// vyresleni car
void lines(int delka)
{
  for (int i = 0; i < delka; i++)
  {
    draw_pixel();
    hide_cursor();
    move_right();
    move_right();
    show_cursor();
  }
}
// zobrazeni v terminalu
void lines_draw(int delka)
{
  set_green_color();
  lines(delka);

  move_to(2, 5);

  set_blue_color();
  lines(delka);

  move_to(4, 6);

  set_red_color();
  lines(delka);

  move_to(6, 1);

  set_white_color();
  lines(delka);

  move_to(5, 20);

  set_black_color();
  lines(delka);

  move_to(6, 1);

  set_yellow_color();
  lines(delka);
}
// vykresleni schodu
void stairs_down(int delka)
{

  for (int i = 0; i < delka; i++)
  {
    draw_pixel();
    move_right();
    draw_pixel();
    move_down();
    draw_pixel();
    move_right();
    draw_pixel();
  }
}
// zobrazeni v terminalu
void stairs_draw(int delka)
{
  set_yellow_color();
  stairs_down(delka);

  move_to(7, 1);
  set_black_color();
  stairs_down(delka);

  move_to(10, 13);
  set_white_color();
  stairs_down(delka);

  move_to(3, 13);
  set_red_color();
  stairs_down(delka);
}
//vykresleni kytky do terminalu
void kytka(int vyska, int sirka)
{
  int stonek = vyska * 0.75;

  set_green_color();
  for (int i = 0; i < stonek; i++)
  {
    if (i == 1)
    {
      move_right();
      draw_pixel();
      move_left();
    }
    if (i == 4)
    {
      move_left();
      draw_pixel();
      move_right();
    }

    draw_pixel();
    move_up();
  }

  set_red_color();
  int do_prava = sirka * 0.5 - 1;
  int do_leva = sirka - do_prava - 1;
  for (int j = 0; j < vyska - stonek - 1; j++)
  {
    draw_pixel();
    move_right();
    for (int i = 0; i <= do_prava; i++)
    {
      draw_pixel();
      move_right();
    }

    for (int i = 0; i <= do_prava; i++)
    {
      move_left();
    }

    for (int i = 0; i <= do_leva; i++)
    {
      draw_pixel();
      move_left();
    }

    for (int i = 0; i <= do_leva - 1; i++)
    {
      move_right();
    }
    if (j == vyska - stonek)
    {
      set_white_color();
      draw_pixel();
    }
    else
    {
      move_up();
    }
  }
  set_white_color();
  draw_pixel();
}
//brava louky
void set_light_green_color()
{
  printf("\x1b[102m");
  flush();
}
//vykresleni louky do terminalu
void louka(int radek, int sloupec, int kvetin, int vyska, int sirka)
{
  set_light_green_color();
  for (int i = 0; i < radek; i++)
  {
    for (int j = 0; j < sloupec; j++)
    {
      draw_pixel();
      move_right();
    }
    move_to(i, 0);
  }
  int x = vyska + 1;
  int y = sirka + 1;

  for (int i = 0; i < kvetin; i++)
  {
    move_to(x, y);
    kytka(vyska, sirka);
    y += sirka + 2;

    if (y >= sloupec - sirka)
    {
      y = sirka + 1;
      x += x;
    }
    if (x >= radek)
    {
      break;
    }
  }
}
//ctverec pro semafor
void ctverec()
{
  show_cursor();
  for (int i = 0; i < 2; i++)
  {
    for (int j = 0; j <= 8; j++)
    {
      draw_pixel();
      move_right();
    }
    move_down();
    move_left();
    for (int j = 0; j <= 8; j++)
    {
      draw_pixel();
      move_left();
    }
    move_down();
    move_right();
  }
  hide_cursor();
}
//vykresleni semaforu do terminalu
void semafor(int barva)
{
  set_black_color();
  for (int i = 0; i < 14; i++)
  {
    for (int j = 0; j <= 16; j++)
    {
      draw_pixel();
      move_right();
    }
    move_down();
    move_left();
    for (int j = 0; j <= 16; j++)
    {
      draw_pixel();
      move_left();
    }
    move_down();
    move_right();
  }

  switch (barva)
  {
  case 0:
    set_white_color();
    move_to(4, 24);
    ctverec();
    move_to(12, 24);
    set_white_color();
    ctverec();

    move_to(20, 24);
    set_white_color();
    ctverec();
    break;
  case 1:
    set_red_color();
    move_to(4, 24);
    ctverec();
    move_to(12, 24);
    set_white_color();
    ctverec();

    move_to(20, 24);
    set_white_color();
    ctverec();

    break;

  case 2:
    set_red_color();
    move_to(4, 24);
    ctverec();
    move_to(12, 24);
    set_yellow_color();
    ctverec();

    move_to(20, 24);
    set_white_color();
    ctverec();
    break;
  case 3:
    set_white_color();
    move_to(4, 24);
    ctverec();
    move_to(12, 24);
    set_white_color();
    ctverec();

    move_to(20, 24);
    set_green_color();
    ctverec();
    break;

  case 4:
    set_white_color();
    move_to(4, 24);
    ctverec();
    move_to(12, 24);
    set_yellow_color();
    ctverec();

    move_to(20, 24);
    set_white_color();
    ctverec();
    break;
  }
}
//animace semaforu
void animace()
{
  int stav = 1;
  while (true)
  {
    move_to(1, 20);
    semafor(stav);
    animate();

    stav += 1;
    if (stav > 4)
    {
      stav = 1;
      clear_screen();
    }
    clear_screen();
  }
}

int main()
{
  // Keep this line here
  clear_screen();

  // Load the input - what should be drawn.
  int drawing = 0;
  scanf("%d", &drawing);
  //   scanf("%d", &drawing);
  int delka = 10;

  int sirka = 8;
  int vyska = 10;

  switch (drawing)
  {
  case 0:
    lines_draw(delka);
    break;

  case 1:
    stairs_draw(delka);
    break;

  case 2:
    move_to(20, 20);
    kytka(vyska, sirka);
    break;

  case 3:
    louka(30, 100, 10, 10, 10);
    break;

  case 4:
    animace();
    break;

  default:
    break;
  }

  // Keep this line here
  end_drawing();

  return 0;
}