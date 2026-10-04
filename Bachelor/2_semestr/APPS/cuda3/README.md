# **Zadání**

Manipulace s více obrázky, alfa kanál, manuální alokace paměti, animace (opakovaná manipulace s obrázkem).

0. Rozšiřte si použití struktury CudaImg tak, aby byl použit třetí rozměr m_size pro informaci o počtu barev - 1, 3, 4 pro B/W, RGB a RGBA. Je také užitečné napsat si funkci, či metodu, která nastaví všechny instanční proměnné struktury CudaImg z objektu třídy Mat. Např. void set_cuda_img( cv::Mat &t_cv_img, CudaImg &t_cuda_img ) { ... }, ale možností je více.

1. Na základě kernelu a funkce z příkladu cuda5 si upravte stávající, nebo vytvořte další funkci a kernel pro vkládání obrázku, který není RGBA, ale jen RGB a alfa kanál bude zadán jako další parametr. Funkce i kernel tak bude mít 4 parametry:

```c++
void cu_insert_rgb_image( CudaImg t_big_cuda_pic, CudaImg t_small_cuda_pic, int2 t_position, uint8_t t_alpha ).
```

2. Níže je funkce implementující algoritmus pro obecnou rotaci obrázku. Přepište si implementaci pro GPU (CUDA), aby rotace obrázku proběhla v kernelu (smyčky for pro souřadnice x a y nahradí mřížka).

Do funkce cu_run_rotate bude vstupovat dvojice obrázků (originál a výsledný otočený) a úhel rotace. Do kernelu bude vstupovat se dvěma obrázky vypočtená hodnota sin a cos, aby se nemusela počítat v každém vlákně.

Upravte si kód, aby zvládl obrázky RGB i RGBA.

Nikdy nerotujte už jednou otočený obrázek. Vždy otáčejte originál!

3. První animace je “změna nálady”, viz video. Použijte dva obrázky smile-pos a smile-neg a funkci z bodu 1.

Oba obrázky se budou vkládat do prázdného obrázku pro postupně rostoucí a pak klesající alpha-level:

```c++
opakuj()
  vloz_obrazek( pozadi, smile1, 255-alpha-level );
  vloz_obrazek( pozadi, smile2, alpha-level );
```

4. Mikropočítače mají integrovaný obvod “watchdog”, fakulta má watchtiget. Pro implementaci použijte jen opakované vkládání obrázku. Obrázky tigra jsou v archivu.

5. Pro ověření rotace obrázku implementujte větrné mlýny. Vyzkoušejte různé rychlosti a směry rotoru. Ověřte, že rotace již rotovaného obrázku není dobrý nápad.

Pokud se Vám vzdáleně špatně vyvíjí “animace”, je možno si zaznamenat obrázky jako video:

```c++
  cv::Mat obrazek(...);
  cv::VideoWriter l_video( "video.mkv", cv::VideoWriter::fourcc( 'H', '2', '6', '4' ), 25, obrazek.size() );

...
...
  cv::imshow( "Titulek", obrazek );
  l_video.write( obrazek );
```

Oba řádky navíc (vytvoření videa a zápis snímku) je pak snadné v případě potřeby zakomentovat.

Implementace rotace v OpenCV.

```c++
void cv_run_rotate( cv::Mat &t_cv_img_orig, cv::Mat &t_cv_img_rotate, float t_angle )
{
 float t_sin = sinf( t_angle );
 float t_cos = cosf( t_angle );

 for ( int l_rotate_x = 0; l_rotate_x < t_cv_img_rotate.cols; l_rotate_x++ )
 {
     for ( int l_rotate_y = 0; l_rotate_y < t_cv_img_rotate.rows; l_rotate_y++ )
     {
         // recalculation from image coordinates to centerpoint coordinates
         int l_crotate_x = l_rotate_x - t_cv_img_rotate.cols / 2;
         int l_crotate_y = l_rotate_y - t_cv_img_rotate.rows / 2;

         // position in orig image
         float l_corig_x = t_cos * l_crotate_x - t_sin * l_crotate_y;
         float l_corig_y = t_sin * l_crotate_x + t_cos * l_crotate_y;
         // recalculation from centerpoint coordinates to image coordinates
         int l_orig_x = l_corig_x + t_cv_img_orig.cols / 2;
         int l_orig_y = l_corig_y + t_cv_img_orig.rows / 2;
         // out of orig image?
         if ( l_orig_x < 0 || l_orig_x >= t_cv_img_orig.cols ) continue;
         if ( l_orig_y < 0 || l_orig_y >= t_cv_img_orig.rows ) continue;

         t_cv_img_rotate.at<cv::Vec3b>( l_rotate_y, l_rotate_x ) = t_cv_img_orig.at<cv::Vec3b>( l_orig_y, l_orig_x );
     }
 }
}
```