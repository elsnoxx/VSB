-- Uloha 1 
-- Definujeme datovy vlastny jftyp "Copmapny", ktory ma tri fieldny 
--name of company string reprezentujuci meno company 
--numberOfEmployess je int reprezentujuci počet employes v company 
--ownerOf je list [Company] reprezentujuci firmy ktore su vlastnene current company
-- vytvarame vlastny datovy typ s nazvom company vdaka našmu klucovemu slovu data kde špecifikujem ze sa bude jednat o datovy typ company ktory bude mat dva fields 
data Company = Company {
  nameOfCompany :: String, -- nameof campany bude mat datovy typ string 
  numberOfEmployees :: Int, -- datovy typ je int teda celočiselna hodnota 
  ownerOf :: [Company] -- owner of je teda list company definujeme ho ako vlastnika 
} deriving Show -- co nam robi funckia deriving show robi to ze z binarneho formatu dat dokaze cloveku umoznit čitatelnost sttringoveho formatu teda human avaiable to see e

-- vytvorime si tri inštancie našho datoveho formatu 
-- creatujeme instanciu Company type named company1 ma mene copmany1 420 zamestnancov a empty list ([]) teda neownuje ziadne ine companies
company1 :: Company
company1 = Company "Company1" 420 []  -- to znamena ze formou vystupiu bude len a len naš prazdny list nič viac nič menej 


-- rovnako creatujeme aj company2 -- ani ta inštancia nevlastni ziadne ine companies 
company2 :: Company
company2 = Company "Company2" 69000 []
-- vytvar
-- tu vytvarame inštanciu s menom parentcompany - 10,000,000 employees a list companies, ktore vlastni ten indikuje nam to ze vlastni obe vyššie dekladovane inšancie 
parentCompany :: Company
parentCompany = Company "Parent to all companies" 10000000 [company1, company2]
-- u parent company mozeme sledovať ze sas bude jednať o vlatnika teda bude feeelupnuty list company 
-- Implementačne nutnosti pre ulohu 2 a 3 

-- definujeme data type Component ktory sa sklada z troch konštruktorov 

-- sledujeme samostnu implementaciu našu data componentu co to znamena je len jednoducha vec 
data Component = TextBox {name :: String, text :: String} -- reprezentuje text box s name a textom oba su stringove formaty 
               | Button {name :: String, value :: String} -- button je reprezentovany namom a valuom 
               | Container {name :: String, children :: [Component]} deriving Show -- reprezentuje container s namom a listom children komponentov 

-- mozšeme sledovať ze sme si vytvorili tri containeri first one je jednoducho textbox ktory bude reprezentovany name formatu string a textom formatu opätovne stringovskeho 
gui :: Component
gui = Container "My App" [ -- highest class respektive top container s menom My app ktory obsahuje dalsi container s menom my app 
    Container "Menu" [ -- menu container bude obsahovať 
        Button "btn_new" "New", -- 3 butttny prvy button  bude s znenim btn_new ktroy bude reprezentovať teda new 
        Button "btn_open" "Open", 
        Button "btn_close" "Close"],
    Container "Body" [TextBox "textbox_1" "Some text goes here"], -- mozeme vidiet ze obsahuje nijaky format pod texti 
    Container "Footer" []]

-- vytvarame inštaciu Compontnt s nazvom gui
-- reprezentuje gui štruktúru s top-level container s menom "My App" obsahujucu tri sub-containers ("Menu," "Body," and "Footer")
-- kazdy sub container ma svoje vlastne komponenty buttons / text boxes 

-- Uloha 2 
-- co nam v našom pripade robia naše anonymus znaky robia velmi jednoduchu vec a tou je vlastne len to ze je jedno čo bude na vstupe 
-- co to znaci hovori to len o jednej veci a to ze na vstupe bude nijaka forma componentu a vystupom bude jej kalkul teda kalkulacia 
-- jedna sa o funkciu ktora ako mozeme sledovať podla headru zoberie ako vstup compenent a vystup je nijaka celočiselna hodnota počtu vyskytu 
countAllComponents :: Component -> Int -- vyuzivame realizaciu prostrednictvom pattern matchingu 
countAllComponents (TextBox _ _) = 1 -- sledujeme či teda naš input je textbox alebo buton v oboch pripadoch bude count inkriminovany o 1 nakolko pretoze textBox alebo button itself su brane ako komponenty 
countAllComponents (Button _ _) = 1 -- rovnaka logika ako na riadku 50 
countAllComponents (Container _ children) = 1 + sum (map countAllComponents children)
-- v realizacii nam pomahahaju len a len naše anonymus markre 
-- klasicke vyuzitie rekurzie na countovanie komponentov v ramci children listu 
--mapuje countAllComponents pre kazdy element v liste produkuje list of counts 
-- nasledne len sum funkcia adds upne count 
-- +1 znamene len addnutie countu pre current container ktory je countovany ako component 
-- priklady vstupu 

-- countAllComponents (TextBox "exampleTextBox" "Some text")
-- countAllComponents (Button "exampleButton" "Click me")
-- countAllComponents (Container "exampleContainer" [TextBox "text1" "Text", Button "btn1" "Click"])


-- Ukol 3

-- priklady inputy pre funckiu removeEmptyContainers

--removeEmptyContainers (TextBox "textName" "textValue")
--removeEmptyContainers (Button "buttonName" "buttonValue")
--removeEmptyContainers (Container "containerName" [TextBox "child1" "value1", Button "child2" "value2"])
--removeEmptyContainers (Container "emptyContainer" [])

-- tato funkcia odstranuje prazdne containeri z hierarchie componentov 
-- podla headeru ako input je brany container a vystupom je modifnuty container
-- pre textbox a button funkcia vrati rovnaky komponent pretoze su to tzv leaf komponenty ktore nemoze byt prazdne 
-- pre container sleduje cu containerr nie je przadny cez isEmpty function
-- pokial nie je prazdna reekurzivne vola removeEmptyContainers na kazdy child component, filtruje prazdne 
-- vytvara nove Container pre non empty child 
removeEmptyContainers :: Component -> Component -- vstupopom je teda nijaka forma componentu ktory je na vstupe vystupom bude teda len component na vystupe 
removeEmptyContainers(TextBox name value) = TextBox name value
removeEmptyContainers(Button name value) = Button name value
removeEmptyContainers (Container name children) = 
    if not (isEmpty (Container name children)) 
    then Container name (filter (not . isEmpty) (map removeEmptyContainers children))
    else Container name [] -- co znači to len a len jednoduchu vec tou je vystupom je prazdny list 

-- pomocna funkcia 
--pokial je input Container s empty listom of children 
-- povazujeme ho za empty a funkcia returnuje True 
-- pre kazdy iny type of component funkcia returnuje false indikuje none empty 
isEmpty :: Component -> Bool -- naša pomocna funckia teda podla headeru berie to ze vstupom sa teda stava nijaky ten componet a vystupom je teda bool hodnota vyskytu 
isEmpty (Container _ []) = True -- teda pokial je na vstupe container kedy ignorujeme inu hodnotu okrem containeru vdaka našmu anonymus markru 
isEmpty _ = False -- naš base case kedy vystupom je teda false 