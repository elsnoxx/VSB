
**Camera — Použití, výpočty a OOP vzory v projektu**

- **Role:** Kamera drží světovou pozici (`eye`), směr pohledu (`target`) a `up` vektor. Počítá view a projection matice, které používají shadery k vykreslení scény.
- **Kde:** `Camera/Camera.h`, `Camera/Camera.cpp`.

**Kdy a kdo kameru aktualizuje**
- `InputManager` (nebo `Scene::update`) předává myší vypočítané delty do `Camera::updateOrientation`.
- `Scene::update` volá pohybové metody (`forward`, `left`, ...) podle stisku kláves.
- `Camera::updateScreenSize` se volá při změně velikosti okna (např. v `Application::updateViewport`).
- `ShaderProgram` se přihlásí jako observer kamery (`camera->attach(observer)`) a při notifikaci aktualizuje uniformy (`viewMatrix`, `projectionMatrix`, `viewPosition`).

**Hlavní výpočty a vzorce**
- View matice: použití `glm::lookAt(eye, eye + target, up)`.
- Projekční matice: použití `glm::perspective(fov, aspect, near, far)`.
- Aktualizace směrového vektoru z úhlů (sférické souřadnice):
	- alpha (yaw) a fi (pitch) se mění podle myši
	- direction.x = cos(fi) * sin(alpha)
	- direction.y = sin(fi)
	- direction.z = -cos(fi) * cos(alpha)
	- potom `target = normalize(direction)`.
- Pohyb (forward/back/left/right):
	- forward/back: `eye += normalize(target) * speed * dt`
	- left/right: použít `cross(target, up)` pro pravý vektor a po něm posunout.
- Adaptivní FOV: interpolace mezi min/max FOV podle poměru stran (v `updateScreenSize`).

**Použité OOP / návrhové vzory**
- Observer pattern
	- `Camera` dědí z `Subject`.
	- `ShaderProgram` a `HeadLight` implementují `Observer` a dostávají update voláním `update(ObservableSubjects::SCamera)`.
	- Výhodou je, že při změně kamery se automaticky aktualizují všechny přihlášené objekty (shadery, headlight).
- Non-owning bindings
	- `InputManager::bindCamera(Camera*)` a `ScreenManager::bindInput(InputManager*)` používají ne-vlastnické ukazatele — je důležité zajistit životnost objektů.
- Factory / Cache
	- `ShaderFactory` a `ModelManager` spravují sdílené zdroje (cache), shadery se opakovaně nepřekládají.
- Kompozit transformací
	- `Transform` / `TransformNode` tvoří strom transformací pro objekty.

**Průběh (control flow) — stručně**
1. `Application::initialization()` vytvoří okno a zavolá `screenManager.init()`.
2. `ScreenManager::init()` vytvoří scény (`SceneFactory`) a pro počáteční scénu zaváže `input->bindCamera(cam)` a `input->bindScene(scene)`.
3. Hlavní smyčka (`Application::run`) volá každý snímek `screenManager.update(dt, input)` a `screenManager.draw()`.
4. `Scene::update` čte `input.getMouseDeltaAndReset(dt)` a volá `camera->updateOrientation(delta, dt)`.
5. `ShaderProgram` je observer kamery a při notifikaci nahraje uniformy do shaderu.

**Kde hledat chyby, když kamera "neposílá" data do shaderů / světla se neaktualizují**
- Uniform mismatch: zkontrolujte jména uniform ve shaderech (např. `viewPosition`) a v `ShaderProgram::update()`.
- Shader není připojen ke kameře: `Scene::addObject()` volá `shader->attachCamera(camera)`; pokud `getShader()` vrátí `nullptr`, připojení se nestane.
- Input priming: `InputManager::firstMouse` ignoruje první událost — je nutné při přepnutí scény primovat `lastMousePos` (to je řešeno v `Application`).
- Pořadí inicializací: ujistěte se, že `screenManager.bindInput(&input)` je voláno před `screenManager.init()`.

**Doporučení pro udržitelnost**
- Centralizovat názvy uniform do jedné hlavičky nebo helperu, aby se předešlo mismatches.
- Přidat kontrolní logy (debug prints) při bind/attach operacích (pomůže ladit problémy s neaktualizujícími se shadery).
- Zvažte silnější model vlastnictví (smart pointery) pro objekty, které sdílejí životnost.

--
Soubor: `Camera/Camera.h`, `Camera/Camera.cpp` — dokumentace vytvořena automaticky z analýzy kódu.

