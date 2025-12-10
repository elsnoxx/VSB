**Project Overview**
- **Purpose:**: Malá OpenGL aplikace (studentský projekt) pro vykreslování 3D scén, správu shaderů, modelů a osvětlení.
- **Language / Tools:**: C++17, OpenGL (GLFW + GLEW), GLM pro matematiku.

**Build & Run (high level)**
- **Open in:**: Visual Studio (s řešením `Projekt.sln`).
- **Dependencies:**: `glew`, `glfw`, `glu32` (viz `Projekt.vcxproj` linkování). Přidejte knihovny do projektu a nastavte include/lib cesty podle lokální instalace.
- **Run:**: Sestavte řešení a spusťte `Projekt.exe` v konfiguraci `Debug` nebo `Release`.

**Project Structure (přehled)**
- **`Application.*`**: hlavní třída aplikace — inicializace OpenGL, hlavní smyčka, zpracování událostí a přepínání scén.
- **`Callbacks.*`**: C-style callbacky pro GLFW, které směrují události do instance `Application` (klávesy, kurzor, myš, error).
- **`Config.*`**: globální konstanty a nastavení (velikost okna, cesty ke shaderům, FOV, vektory apod.).
- **`Scene/`**: správa scén, továrna scén, `ScreenManager` — přepínání mezi scénami, vykreslení aktuální scény.
- **`Shader/`**: načítání a kompilace shaderů, správa `ShaderProgram` a zdrojů shaderů (`ShaderSources/`).
- **`ModelObject/`**: reprezentace objektů a modelů, `ModelManager`, textury a assety.
- **`Transform/`**: implementace transformací (translation, rotation, scale, bezier), uzly transformací.
- **`Camera/`**: kamera, pohyb a výpočet projekcí a pohledových matic.
- **`Input/`**: `InputManager` — abstrakce vstupu a režimu zpracování kláves.
- **`Light/`**: třídy osvětlení (directional, point, spot, headlight) a `LightManager`.
- **`Observer/`**: jednoduchá implementace observer patternu (`Subject`, `Observer`).

**Core Components — jak fungují (detailněji)**
- **`Application`**: : Zodpovídá za inicializaci GLFW/GLEW (`initialization()`), nastavení viewportu a hlavní render loop (`run()`). V okně ukládá ukazatel na sebe pomocí `glfwSetWindowUserPointer(window, this)`; callbacky pak tento ukazatel čtou a volají metody instance (např. `handleMouseClick`, `updateViewport`). Třída drží `ScreenManager`, přes který komunikuje se scénami.
- **`Callbacks`**: : Definovány jako C-funkce vyžadované GLFW. V callbacku kláves přeposílají události do `Application::input` (volání `app->input.onKey(...)`) a do aplikace (např. přepínání scény nebo změna FOV). Mouse/ cursor callbacky volají `handleMouseClick` a `updateViewport`.
- **`ScreenManager` & `Scene`**: : `ScreenManager` drží kolekci scén (pravděpodobně vytvořených přes `SceneFactory`) a poskytuje metody `init()`, `switchTo(index)` a `getCurrentScene()`. Scény implementují vykreslování, logiku, a metody pro pickování objektů (`pickAtCursor`), které Application používá pro interakci myší.
- **`Shader` systém**: : `ShaderFactory` a `ShaderProgram` načítají a kompilují vertex/fragment shadery z `Shader/ShaderSources/`, abstrakce pro shader typy (phong, lambert, textured, skybox...). `ShaderProgram` spravuje uniformy a aktivaci shaderu pro kreslení.
- **`Model` a `ModelManager`**: : `Model` (a `DrawableObject`) reprezentují geometrii + materiály; `ModelManager` centralizuje nahrávání a přístup k modelům a texturám. Assets jsou v `ModelObject/assets`.
- **`Transform`**: : Třídy pro charakter transformací objektů (kompozice transformací do stromu přes `TransformNode`). Každý objekt má transformaci složenou z translation/rotation/scale či komplexnějších křivek (Bezier).
- **`Camera` & FOV`**: : Camera drží pozici a orientaci, generuje view a projection matice. FOV se mění přes `Application::updateFOV()` vyvolané z klávesových callbacků.
- **`InputManager`**: : Abstrakuje zpracování stavu kláves a potenciálně myši — `Application` používá `input` pro reakce.
- **`LightManager`**: : Spravuje kolekci světel a jejich předávání do shaderů (uniformy). Různé typy světel dědí (`DirectionalLight`, `PointLight`, `SpotLight`, `HeadLight`) nebo implementují společné rozhraní.
- **`Observer` pattern**: : Jednoduché `Subject/Observer` třídy pro notifikace mezi komponentami (viz `Observer/`). Použitelné pro event-driven updaty scén nebo UI.

**OOP návrhové poznámky (kde a proč)**
- **Single Responsibility:**: `Application` řeší lifecycle a události; `ScreenManager` přepínání scén; `ShaderProgram` pouze práci se shadery; `ModelManager` pouze modely — separace povinností je dodržena.
- **Encapsulation:**: Každá část (shadery, modely, světla) nabízí veřejné API a skryté internály (např. kompilace shaderu interně).
- **Composition over inheritance:**: Transformy a Drawable objekty jsou komponovány z menších částí (transform kroky, material, mesh) místo těžkého dědění.
- **Factory pattern:**: `SceneFactory` a `ShaderFactory` vytvářejí instance a centralizují konfiguraci (usnadňuje rozšíření).
- **Manager classes:**: `ScreenManager`, `ModelManager`, `LightManager` slouží jako registry a lifecycle manažeři.

**Praktické implementační poznámky / tipy pro úpravy**
- **Inicializace:**: `Application::initialization()` nastavuje GLFW hinty, vytváří okno, volá `glewInit()`, nastavuje `glViewport` a callbacky.
- **Callback flow:**: GLFW volá `callback*` funkce, které přečtou `Application` ukazatel z okna a přepošlou událost do instančních metod. To umožňuje OOP styl práce i s C-style API.
- **Picking / interakce:**: V `Application::handleMouseClick` se získá aktuální scéna (`getCurrentScene()`), zavolá se `pickAtCursor(x,y,&worldPos)` a podle výsledku se provede akce (např. výběr objektu nebo update kamery). Tato separace logiky pickování do scény je dobrý přístup.
- **Rozšíření:**: Přidání nového shaderu: přidejte soubory do `Shader/ShaderSources/` a zaregistrujte typ v `ShaderFactory`. Přidání nového objektu: implementujte `Model`/`DrawableObject` a nahrát přes `ModelManager`.

**Where to look first (rychlá navigace)**
- **Start:**: `Application.cpp` / `Application.h` — entry point, inicializace a hlavní smyčka.
- **Input / Events:**: `Callbacks.*` a `Input/InputManager.*`.
- **Scene logic:**: `Scene/Scene.cpp`, `SceneFactory.cpp`, `ScreenManager.cpp`.
- **Rendering pipeline:**: `Shader/ShaderProgram.*`, `ShaderFactory.*`, `ShaderSources/`.

**Next steps / doporučení**
- **Přidat UML diagram:**: Můžu vytvořit jednoduchý class diagram (textový nebo SVG) pro vybrané třídy.
- **Rozšířit README:**: Přidat konkrétní build kroky pro různá prostředí, seznam externích závislostí a nastavení `Include/Lib` cest.
- **Tests / run script:**: Můžu přidat PowerShell skript pro automatické sestavení ve Windows.

Pokud chcete, mohu README upravit dále (přidat UML, build kroky, nebo skripty). Napište, co preferujete.
