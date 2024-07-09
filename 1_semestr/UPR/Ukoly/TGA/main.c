#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h>
// funkce pro vypis eroru
void errory(char *hlaska)
{
    printf("%s\n", hlaska);
    exit(1);
}
typedef unsigned char byte;

typedef struct
{
    byte id_length;
    byte color_map_type;
    byte image_type;
    byte color_map[5];
    byte x_origin[2];
    byte y_origin[2];
    byte width[2];
    byte height[2];
    byte depth;
    byte descriptor;
} TGAHeader;

typedef struct
{
    byte blue;
    byte green;
    byte red;
} Pixel;

typedef struct
{
    char letter;
    TGAHeader header;
    Pixel *pixel_img;
    int width;
    int height;
} TGAFontImg;

// nacteni pixelu
Pixel *load_pixels(TGAHeader header, FILE *file)
{
    int width = 0;
    int height = 0;

    memcpy(&width, header.width, 2);
    memcpy(&height, header.height, 2);

    Pixel *pixels = (Pixel *)malloc(sizeof(Pixel) * width * height);
    assert(fread(pixels, sizeof(Pixel) * width * height, 1, file) == 1);
    return pixels;
}

TGAFontImg *load_fonts(char *fonts)
{
    TGAFontImg *font_img = (TGAFontImg *)malloc(26 * sizeof(TGAFontImg));
    int height = 0;
    int width = 0;

    for (int i = 0; i < 26; i++)
    {
        char path[100];

        sprintf(path, "%s/%c.tga", fonts, 'A' + i);

        FILE *file = fopen(path, "rb");
        if (!file)
        {
            errory("Could not open font image.\n");
        }

        font_img[i].letter = 'A' + i;
        font_img[i].header = (TGAHeader){}; // Initialize header to zero
        assert(fread(&font_img[i].header, sizeof(TGAHeader), 1, file) == 1);

        memcpy(&height, font_img[i].header.height, 2);
        memcpy(&width, font_img[i].header.width, 2);

        font_img[i].height = height;
        font_img[i].width = width;

        font_img[i].pixel_img = load_pixels(font_img[i].header, file);

        fclose(file);
    }

    return font_img;
}

int vypocet_zacatku(TGAFontImg *font_img, char *radek,int width_main, int spaces){
    int suma = 0 ;
    int cnt = 0;
    for (size_t i = 0; i < strlen(radek); i++)
    {
        if (radek[i] == '\0')
        {
            break;
        }
        for (int j = 0; j < 26; j++)
        {
            if(font_img[j].letter == radek[i]){
                suma += font_img[j].width;
            }
            
        }        
    }
    printf("-----%d-----%d-----%d\n",suma, (width_main - suma - spaces * 10) / 2 ,cnt);
    return ((width_main - suma - spaces * 10) / 2 );
}


void draw(TGAFontImg *font_img, char *radek, int start_row, Pixel *main_pixels, int width_main, int height_main, int spaces)
{
    int spacing = 10;
    int current_col = vypocet_zacatku(font_img, radek, width_main,spaces);
    for (size_t i = 0; i < strlen(radek); i++)
    {
        for (int j = 0; j < 26; j++)
        {
            if (font_img[j].letter == radek[i])
            {
                int start_col = current_col;
                int end_col = current_col + font_img[j].width;

                for (int cur_row = start_row; cur_row < start_row + font_img[j].height && cur_row < height_main; cur_row++)
                {
                    for (int col = start_col; col < end_col && col < width_main; col++)
                    {
                        int main_index = cur_row * width_main + col;
                        int font_index = (cur_row - start_row) * font_img[j].width + (col - start_col);
                        if (font_img[j].pixel_img[font_index].red != 0 || font_img[j].pixel_img[font_index].green != 0 || font_img[j].pixel_img[font_index].blue != 0){
                            main_pixels[main_index] = font_img[j].pixel_img[font_index];
                        }
                    }
                }
                current_col += font_img[j].width;
                break;
            }
            else if (radek[i] == 32 && spaces >= 0)
            {
                current_col += spacing;
                spaces--;
                break;
            }
        }
    }
}
// vypis TGA obrazku
void vypis(TGAHeader input_head, Pixel *pixels, char *output, int height_main, int width_main)
{
    FILE *vystup = NULL;
    vystup = fopen(output, "wb");
    fwrite(&input_head, sizeof(TGAHeader), 1, vystup);
    fwrite(pixels, sizeof(Pixel), height_main * width_main, vystup);
    fclose(vystup);
}

int main(int argc, char *argv[])
{
    // kontroloa vstupnich parametru
    if (argc < 4)
    {
        errory("Wrong parameters");
    }

    char *input = argv[1];
    char *output = argv[2];
    char *fonts = argv[3];
    FILE *vstup = NULL;

    int top = 0;
    int botom = 0;
    char buffer[101];

    // kontrola otevreni vstupniho tga obrazku
    vstup = fopen(input, "rw");
    if (vstup == NULL)
    {
        errory("Could not load image");
    }
    assert(vstup);
    TGAHeader input_head = {};
    assert(fread(&input_head, sizeof(TGAHeader), 1, vstup) == 1);

    int height_main = 0;
    int width_main = 0;
    memcpy(&height_main, input_head.height, 2);
    memcpy(&width_main, input_head.width, 2);
    // printf("height %d, width %d\n", height_main, width_main);

    Pixel *pixels = load_pixels(input_head, vstup);

    TGAFontImg *font_img = load_fonts(fonts);

    scanf("%d %d", &top, &botom);
    int row = 0;
    int botom_row = botom;
    int spaces = 0;
    // printf("input: %s, output %s, fonts %s, top %d, botom %d\n",input,output,fonts,top,botom);
    for (int i = 0; i <= top + botom; i++)
    {
        fgets(buffer, sizeof(buffer), stdin);
        for (size_t a = 0; a < strlen(buffer); a++)
        {
            buffer[a] = toupper(buffer[a]);
            if (buffer[a] == 32)
            {
                spaces++;
            }
        }
        if (i <= top) 
        {
            // printf("%d\n",row + 34);
            draw(font_img, buffer,row-24, pixels, width_main,height_main,spaces);
            
        }else if (i > top)
        {
            row = height_main - 34 * botom_row - 10;
            draw(font_img, buffer,row, pixels, width_main,height_main,spaces);
            botom_row--;
        }
        row += 34;
        // printf("--%d--\n",spaces);
        vypocet_zacatku(font_img, buffer, width_main,spaces);
        spaces = 0;
    }

    vypis(input_head, pixels, output,height_main, width_main);

    for (int i = 0; i < 26; i++)
    {
        free(font_img[i].pixel_img);
    }

    fclose(vstup);
    free(pixels);
    free(font_img);

    
    return 0;
}