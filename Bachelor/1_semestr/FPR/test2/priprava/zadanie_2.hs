

-- uloha 1 vytvorenie datoveho typu Entity
data Entity
  = Point {x :: Double, y :: Double}
  | Circle {stred_x :: Double, stred_y :: Double, polomer :: Int}
  | Box {boxChildren :: [Entity]}
  deriving (Show)

-- úloha 1.2 vytvorenia inštancie pre Entity (Priklad pre Entity)
entita :: Entity
entita =
  Box
    [ Point 7.2 6.5,
      Box
        [ Point 8.1 8.2,
          Circle 3.12 5.1 4,
          Circle 3.5 6.12 7
        ],
      Box
        [ Circle 3.65 8.1 8
        ]
    ]

-- Vzor pre ďalšie úlohy 2 - 3
data Component
  = TextBox {name :: String, text :: String}
  | Button {name :: String, value :: String}
  | Container {name :: String, children :: [Component]}
  deriving (Show)

gui :: Component
gui =
  Container
    "My App"
    [ Container
        "Menu"
        [ Button "btn_new" "New",
          Button "btn_open" "Open",
          Button "btn_close" "Close"
        ],
      Container "Body" [TextBox "textbox_1" "Some text goes here"],
      Container "Footer" []
    ]

-- Uloha 2 spocitat buttons

-- Pokiaľ bude constructor typu Button, tak sa vloží do poľa 1
-- Pokiaľ bude constructor typu TextBox, tak sa vloží do poľa 0
-- Nakonci ak to bude Container, tak sa zvyšok stromu vlozi do poľa za pomoci máp a secte sa pomoci sum

-- príklad inputu countButtons gui
countButtons :: Component -> Int
countButtons (Button _ _) = 1
countButtons (TextBox _ _) = 0
countButtons (Container _ children) = sum (map countButtons children)


-- Funkcia copy Element vezme Component, názov tlačidla a jeho hodnotu a vráti upravenú Component.
-- Ak v zozname children nájde tlačidlo so zhodným názvom, pridá nové tlačidlo pod existujúce.
-- Inak rekurzívne prechádza a upravuje deti kontajnera.

-- priklad inputu copyElement gui "btn_open" "Open Document"
copyElement :: Component -> String -> String -> Component
copyElement (Container name children) buttonName value
  | any (`hasButton` buttonName) children =  -- Kontroluje, či sa v zozname nachádza tlačidlo s daným názvom
      Container name (addNewButton buttonName value children)  -- Pridá nové tlačidlo pod existujúce, ak bolo nájdené
  | otherwise =
      Container name (map (\c -> copyElement c buttonName value) children)  -- Rekurzívne prechádza a upravuje deti kontajnera
copyElement x _ _ = x  -- Pokiaľ sa jedná o tlačidlo (a nie kontajner), vráti sa bez zmien

-- Funkcia hasButton kontroluje, či daný komponent zodpovedá zadanému názvu tlačidla.
-- vstup pre funkciu hasButton gui "Open"

hasButton :: Component -> String -> Bool
-- príklad inputu asButton gui "btn_new"
hasButton (Button name _) btn = name == btn  -- Kontroluje zhodu názvu tlačidla
hasButton _ _ = False  -- Vrací False pro jakoukoli jinou komponentu než tlačítko

-- Funkcia addNewButton pridá nové tlačidlo pod existujúce tlačidlo so zhodným menom v zozname children.
-- addNewButton newButtonName newValue existingComponents
addNewButton :: String -> String -> [Component] -> [Component]
addNewButton btn value [] = [Button (btn ++ "_copy") value] 
-- Ak je zoznam prázdny, pridá nové tlačidlo na začiatok
addNewButton btn value (c : cs) =
  if hasButton c btn
    then c : Button (btn ++ "_copy") value : cs  -- Ak je nájdené tlačidlo so zhodným menom, pridá nové tlačidlo pod neho
    else c : addNewButton btn value cs  -- Inak pokračuje rekurzívne prehľadávať zvyšok zoznamu
