-- Zadani 2
-- Ukol1
data Entity = Point {x :: Double, y :: Double} | Circle {x :: Double, y :: Double, r :: Int} | Contrainer [Entity]

-- UkolZadani2,3
data Article = Text String | Section String [Article] deriving (Show)

myArticle :: Article
myArticle = Section "Document" [
    Section "Introduction" [
        Text "My introduction",
        Section "Notation" [Text "alpha beta gamma"]],
    Section "Methods" [
        Section "Functional Programming" [Text "FPR"],
        Section "Logical Programming" [Text "LPR"]],
    Section "Results" [Text "All is great"]]

-- Ukol2 
-- vrati seznam bloku textu obsazenych v objektech typu Text
-- allTexts myArticle
allTexts :: Article -> [String]
allTexts (Text str) = [str]
allTexts (Section _ list) = concatMap allTexts list

-- Ukol3 (vrati seznam kapitol, ktere ve svem seznamu typu [Article] neobsahuji Text)
-- names myArticle
names :: Article -> [String]
names Text {} = []
names (Section name list) | check list = name : concatMap names list
                          | otherwise = concatMap names list 
                            where
                                check :: [Article] -> Bool
                                check = foldr ((&&) . ch) True 
                                    where
                                        ch Text {} = False
                                        ch Section {} = True


-- Zadani 1
-- ukol1
data Company = Company {name :: String, employees :: Int, ownerOf :: [Company]}

myCompany :: Company
myCompany = Company "Spolecnost" 15000 [Company "Spolecnost2" 50 []]

-- Ukol2 (vrati seznam nazvu kapitol)
-- allSections myArticle
allSections :: Article -> [String]
allSections Text {} = []
allSections (Section name list) = name : concatMap allSections list

-- Ukol3 (vrati nejvyssi pocet vnoreni)
-- articleDepth myArticle
articleDepth :: Article -> Int 
articleDepth Text {} = 0
articleDepth (Section _ list) = 1 + maximum (map articleDepth list)


-- dalsi na procviceni
data FileType = Image | Executable | SourceCode | TextFile deriving (Eq, Show)

data Entry = File {nam :: String, size :: Int, ftype :: FileType}
           | Directory {nam :: String, entries :: [Entry]} deriving (Eq, Show)

root :: Entry
root = Directory "root"
    [
    File "logo.jpg" 5000 Image,
    Directory "classes"
        [
        File "notes-fpr.txt" 200 TextFile,
        File "presentation.jpg" 150 Image,
        File "first_test.hs" 20 SourceCode
        ]
    ]

-- vrátí seznam dvojic: (jméno adresáře, velikost adresáře)
-- directorySizes root
directorySizes :: Entry -> [(String, Int)]
directorySizes File {} = []
directorySizes (Directory name files) = (name, sum [countSize x | x <- files]) : concat [directorySizes x | x <- files]

-- vypise velikos vsech souboru File
countSize :: Entry -> Int
countSize (File _ size _) = size
countSize (Directory _ files) = sum (map countSize files)


-- vypise pocet souboru v adresarove strukture predane jako parametr i v podadresarich
-- countFiles root
countFiles :: Entry -> Int
countFiles File {} = 1
countFiles (Directory _ children) = sum (map countFiles children)


-- dostane jako první parametr název souboru a z adresářové struktury, předané jako druhý parametr, odstraní všechny výskyty souboru s tímto jménem
removeFile :: String -> Entry -> Entry
removeFile str (File name size ftype) = File name size ftype
removeFile str (Directory name files) = Directory name (check str [removeFile str x | x <- files])

check :: String -> [Entry] -> [Entry]
check _ [] = []
check str ((Directory name files):xs) = Directory name files : check str xs
check str ((File name size ftype):xs) | name == str = check str xs
                                      | otherwise = File name size ftype : check str xs



type Soubor = String
type Adr = String
data Adresar = Adresar Soubor Adr [Adresar]

adres::Adresar
adres = Adresar "Soubor1" "adresar1" [Adresar "Soubor2" "adresar2"[], Adresar "Soubor21" "adresar21"[]]




type Jmeno = String
type Rocnik = Int
data Pohlavi = Muz | Zena
data Rodina = Rodina Jmeno Rocnik Pohlavi [Rodina]

malaRodina:: Rodina
malaRodina =
    Rodina "Jan" 1945 Muz
    [
        Rodina "Jiri" 1965 Muz [],
        Rodina "Dana" 1968 Zena
        [
            Rodina "Jan" 1988 Muz[],
            Rodina "Marta" 1955 Zena []
        ]
    ]

pocetBezdetnych :: Rodina -> Int
pocetBezdetnych (Rodina _ _ _ []) =1
pocetBezdetnych (Rodina _ _ _ (x:xs)) = pocetBezdetnych2 (x:xs)

pocetBezdetnych2 :: [Rodina] -> Int
pocetBezdetnych2 [] = 0
pocetBezdetnych2 (x:xs) = pocetBezdetnych x + pocetBezdetnych2 xs

--bezVnoucat :: Rodina -> Int
--bezVnoucat [] = 1
--bezVnoucat(Rodina _ _ _ ) = 1
--bezVnoucat (Rodina _ _ _ ) = bezVnoucat 
