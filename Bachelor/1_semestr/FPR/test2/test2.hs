-- Nadefinuj rekurzivni datovou strukturu COLOR.
-- Instanci tohoto typu je bud: 
        -- Konkretni bravy Black a White
        -- Brava zadana pomoci slozek RGB - tri cisla typu int
        -- Mmix barev tyto barvy jsou ulozeny jako seznam typu COLOR

data Color = Black
            | White
            | RGB Int Int Int
            | Mix [Color]


cerna :: Color
cerna = Black

bila :: Color
bila = White

ruzna :: Color
ruzna = RGB 255 0 0

vice :: Color
vice = Mix [Black, White, RGB 0 255 0]




data Component = TextBox {name :: String, text :: String}
                | Button {name :: String, value :: String} 
                | Container {name :: String, children :: [Component]}



gui :: Component
gui = Container "My App" [
    Container "Menu" [
        Button "btn_new" "New",
        Button "btn_open" "Open",
        Button "no_open" "Open",
        Button "btn_close" "Close"],
    Container "Body" [ TextBox "btn-1" "Some text goes here"],
    Container "Body" [ TextBox "text_box_1" "Some text goes here"],
    Container "Footer" [],
    Container "btn-11" []]


-- ukol 2 
-- implemetujte funkci ktera v predane datove strukture najde vsechny jmena komponent, ktera zacinaji na retezec predany jako druhy parametr
-- startWithName gui "btn"

-- kontrola zdali se tam vyskytuje prefix
startsWith :: String -> String -> Bool
startsWith [] _ = True
startsWith _ [] = False
startsWith (x:xs) (y:ys) = x == y && startsWith xs ys


-- hlavni funkce
startWithName :: Component -> String -> [String]
-- kontrola zdali button name zacina na predany retexec
startWithName (Button name _) prefix = [name | startsWith prefix name]
-- u textbox nepotrebuji nic jelikoz hledam jen v nazvech button
startWithName (TextBox name _) prefix = [name | startsWith prefix name]
-- zanoreni dale do kontejneru a rekurzi volani pro dalsi zanoreni
startWithName (Container _ children) prefix = concatMap (`startWithName` prefix) children



-- ukol 3 
-- dostane jako parametr datovou strukturu, odstrani vsechny vnitrni kontejnery
-- vnitrni kontejner je takovy, ktery je ulozen v jinem kontejneru
-- jinymi slovy, muze zustat pouze kontejner na nejvyssi urovni

-- noContainers :: Component -> Component
-- noContainers 
