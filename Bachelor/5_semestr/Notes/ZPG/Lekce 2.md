
## Info o funkcích
- VBO - buffer s indexaci, nastavi radu floutu 3 pro
- VAO - navod jak ma graficka karta cist 
- projit si a kouknout se na ty prikazy, popsat si a napsat si co jednotlive dela a jak to probiha
- prvni 3f zanci pozici ty druhe 3f znaci normala barvy --> to dava do hromady jeden vrchol, tak to jde potom tak samo dale

## Poznamky
- matice se nasobi radek x sloupec, viz prednaska 3, projit si to 
 - projit si lekci bude chtit odvodit rotacu s vyuzitim rotace souradne soustavy --> prednaska 3
 - afinni transformace 
	 - jaky je vztah mezi obema maticemi (transpozice matice)?  
	 - pozor na variantu kdy je dob reprezentovan radkovym vektorem, defoultne budeme pouzivat se sloupcovim vektorem, opengl ho pouziva taky defaultne ale jde prepnout
	 - problem s tim ze prvne se musi nasobit a pak pricit posun
	 - nelze matice skladat a pocitat dopredu
	
- nasobeni matic neni komutativni, matice nelze prehazovat AxB != BxA
- rotace v prostu se dela pomoci osy, v 2d lze podle bodu
- Projektivni prostor -> z 2D souradni udelam 3D a z 3D souradnic udelam 4D, nejcasteji volime homogeni souradnici jako 1, 
	- o proni afinniho prostoru ma vyhodu jelikoz nam ubude + coz nam ulehci vypocet, kde vsechny vrcholy vynasobim jen jednou


kdy je projektivni prostor rychlejsi 
- kdyz mame hodne bodu a hodne transformaci vyhrava projektivni prostor, ale kdyz je vynasobim jednou tak pak je nasobou vsechny jednou matici
- 1000 bodu a jednu transformaci, kartezsky system
- 1 bod a 1000  - vyhrava kartezsky 
- nekonecno v kartezske soustave => neda se s nim pocitat
- v projektivni se da pocitat s nekonecem a to pokud mam `[5,6,7,0]` dela to ta nula na konci jelikoz by tam byt nemela jelikoz pri presunu zpet se jim pak deli, 
- vse z kartezkeho se da prevest do projektivniho, ale z projektivniho uz ne vsechny jsem schopen prevest do kartezskoho, jelikoz by dochazelo k deleni nulou