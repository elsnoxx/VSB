-- 1. uloha 
data HTMLDocument -- definicia vlastneho data typu, ma dva vlastne konštruktory Atribute a Tag 
  = Attribute {attributeName :: String, attributeValue :: String} -- ma dva fieldy attributeName attributeValue oba su datoveho typu string 
  | Tag {tagName :: String, tagChildren :: [HTMLDocument]} -- takuje dva fieldy tagname a tagChildren je list HTMLDocument reprezentujuci child elementy  
  deriving (Show) -- premitivne implementovana funkcia z haskellu ktora umozni konverziu values do ludmi vnimatelneho formatu stringu


htmlDocument :: HTMLDocument 
htmlDocument =
  Tag -- tzv outreemost tag obsahuje tag s menom html 
    "html"
    [ Tag -- list v HTML tagu obsahuje dva child tagy 
        "head" -- obsahuje tittle tag s atributmi text setnuty na hello world 
        [ Tag "title" [Attribute "text" "Hello, World!"]
        ],
      Tag
        "body" -- ovsahuje h1 tag s atributmiu class id a style 
        [ Tag "h1" [Attribute "class" "heading", Attribute "id" "main-heading", Attribute "style" "color:blue;"],
          Tag "p" [Attribute "id" "paragraph", Attribute "style" "font-size:16px;", Attribute "text" "This is a sample paragraph."], -- paragraph obsahuje atributy id style a text 
          Tag
            "div" -- obsahuje class atributy a vnoreny ul tag 
            [ Attribute "class" "container",
              Tag
                "ul"
                [ Tag "li" [Attribute "text" "Item 1"],
                  Tag "li" [Attribute "text" "Item 2"],
                  Tag "li" [Attribute "text" "Item 3"]
                ]
            ]
        ]
    ]

-- nutnost implementacie pre funkčnosť prvej a druhej ulohy 
data Component -- definujeme data type vďaka tzv "keywordu" data ma tri konštruktory TextBox Button Container
  = TextBox {name :: String, text :: String} -- ma dva fieldy name text oba su datoveho typu string 
  | Button {name :: String, value :: String} -- opätovne dva fieldy name a value oba typu string 
  | Container {name :: String, children :: [Component]} -- ma dva fieldy name ( string ) a children list componentov 
  deriving (Show) -- creatuje default implementaciu funkcie Show - convertne vulues a type do clovekom viditelnej formy stringu

-- hierarchicka štruktura 
gui :: Component
gui =
  Container -- v našom main containeri mame mame dalsie tri child containeri 
    "My App" -- "highest" container nesie meno my app 
    [ Container -- container s menom menu obsahuje tri button komponenty -- teda simulujeme menu 
        "Menu"
        [ Button "btn_new" "New",
          Button "btn_open" "Open",
          Button "btn_close" "Close"
        ],
      Container "Body" [TextBox "textbox_1" "Some text goes here"], -- obsahuje single textbox s menom texbox_1 a nijaky placeholder text 
      Container "Footer" [] -- dalsi container s menom footer bez children reprezentujuci footer 
    ]

-- uloha 2  

--Funkcia print Paths prechádza štruktúrou Component a vytvára reťazec reprezentujúci cestu ku komponentom so zadaným menom.printPaths :: Component -> String -> String
printPaths :: Component -> String -> [Char]
printPaths (Button name _) str
  | name == str = "/" ++ name -- Vytvorenie cesty pre Button, ak meno zodpovedá zadanému reťazcu.
  | otherwise = "" -- Vracia prázdny reťazec, pokiaľ meno nie je zhodné.
printPaths (TextBox name _) str
  | name == str = "/" ++ name -- Vytvorenie cesty pre TextBox, ak meno zodpovedá zadanému reťazcu.
  | otherwise = "" -- Vracia prázdny reťazec, pokiaľ meno nie je zhodné.
printPaths (Container name children) str
  | any (`containsComponent` str) children -- Ak nejaký child komponent obsahuje hľadané meno,
    =
      "/" ++ name ++ concat [printPaths c str | c <- children] -- vytvorí cestu a prechádza ďalej štruktúrou.
  | otherwise = "" -- Vracia prázdny reťazec, pokiaľ žiadne meno nie je zhodné.

-- Funkcia contains Component kontroluje, či je v komponente obsiahnuté zadané meno.containsComponent
-- priklad input trm2 gui "Menu" 0
containsComponent :: Component -> String -> Bool
containsComponent (Button name _) porov = name == porov --Kontroluje, či meno Buttonu zodpovedá zadanému reťazcu.
containsComponent (TextBox name _) porov = name == porov --Kontroluje, či meno TextBoxu zodpovedá zadanému reťazcu.
containsComponent (Container _ children) porov =
  -- Rekurzívne kontroluje detské komponenty, či obsahujú zadané meno.
  any (`containsComponent` porov) children

-- Celý výraz any (containsComponent porov) children prechádza každý detský komponent
-- v children a aplikuje funkciu containsComponent s argumentom porov.
-- Výsledkom je True, pokiaľ aspoň jeden z child komponentov obsahuje zadané meno porov, inak False.

-- Funkcia removeComponentFromContainerAtIndex odstraňuje komponenty z kontejnera s daným menom na zadanom indexe.
-- input removeComponentFromContainerAtIndex gui "Menu" 1
removeComponentFromContainerAtIndex :: Component -> String -> Int -> Component
removeComponentFromContainerAtIndex (Container name children) containerName index
  | name == containerName = Container name (removeAtIndex children index) -- Ak je meno kontajnera zhodné, odstráni komponent na danom indexe.
  | otherwise = Container name [removeComponentFromContainerAtIndex c containerName index | c <- children] -- Rekurzívne prechádza deti, pokiaľ meno nie je zhodné.
removeComponentFromContainerAtIndex x _ _ = x -- Vracia pôvodný komponent, pokiaľ nie je typu Container.

-- Funkcia remove At Index odstraňuje prvok z daného zoznamu na zadanom indexe.
-- input emoveAtIndex [1, 2, 3, 4, 5] 2
removeAtIndex :: [a] -> Int -> [a]
removeAtIndex [] _ = [] -- ak je zoznam prázdny, vráti prázdny zoznam.
removeAtIndex (x : xs) n
  | n == 0 = xs -- Ak dosiahneme index 0, odstránime prvok.
  | otherwise = x : removeAtIndex xs (n - 1) -- Inak rekurzívne pokračujeme cez zvyšok zoznamu.

-- Funkcia trm2 kontroluje, či je daný kontajner so zadaným menom.
-- priklad inputu trm2 gui "My App" 42
trm2 :: Component -> String -> Int -> Bool
trm2 (Container name _) containerName _ = name == containerName -- Porovnáva meno kontajnera so zadaným menom.
trm2 _ _ _ = False -- Vracia False pre všetky ostatné prípady.