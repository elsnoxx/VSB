
globalni souradny system
rasterizace - prevede vektorovy popis na rastrovy, spocita

matice translace -- posledni ma vketro posunuti


normala je vektor -


A' = M`*`A 
-> inverzni matice

P * V * M * vec4(vp)

naucit phongu osvetlovaci model - 
I = Iara + Suma(Idrd * cos alfa + )
alfa - mezi normalou a vektorem ke zdroji svetla, kresli se s strizkou protoze jsoi normalizovane 
- proc musi byt normalizovane - dot produckt muzeme pouzit, doct prodoct je skalarni soucit, tzv vynasobim vsechny mezi sebou nebo jako delka obou vektoru krat cosinus alfa

proc se libi cos ?  -> cos 0 = 1 

(1,2,3) / w != 0

skalarni soucit -> doc(n,t) = 0 => vektroy jsou na sebe kolme, cos 90 == 0

matice se nasobi radek krat sloupec a musi byt stejne hodnoty
normala a tecna jsou kolme musi zustat 0, zapisuje se to v paticovem tvaru ale jinak se jedna o skalarni soucin

```c++
normal = transpose(inverse(mat3(modelMatris))) * normal
```

mit test pro doc product, aby nedochazelo k prenosu svetla na druhrou stranu