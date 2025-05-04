# Spuštění a vytvoření APK

## 1. Instalace závislostí
```sh
npm install
```

## 2. Spuštění aplikace v prohlížeči (vývoj)
```sh
npm start
```

## 3. Build produkční verze
```sh
npm run build
```

## 4. Převod do nativní aplikace (APK) pomocí Capacitor

### a) Inicializace Capacitor (pouze jednou)
```sh
npx cap init
```

### b) Přidání Android platformy (pouze jednou)
```sh
npx cap add android
```

### c) Zkopírování buildu do Android projektu
```sh
npx cap copy
```

### d) Otevření v Android Studiu
```sh
npx cap open android
```

### e) Vytvoření APK v Android Studiu
- V Android Studiu klikni na **Build > Build APK(s)**
- Najdeš soubor `app-debug.apk` ve složce `android/app/build/outputs/apk/debug/`

---

**Poznámka:**  
Pro testování na mobilu musí být zařízení ve stejné síti jako PC, nebo použij APK.