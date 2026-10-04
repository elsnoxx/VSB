# **Vize**

## Proč?

Zákazník chce vytvořit platformu pro centralizovanou správu zařízení, kde bude možné přidávat zařízení, spravovat je podle sériových čísel a přiřazovat je k místům podle jejich umístění. Taková aplikace bude sloužit jako jednotný nástroj pro správu zařízení a jako systém pro evidenci aktiv a kompletní inventář všech zařízení.

## Co?
Aplikace bude obsahovat:
- Správu zařízení (přidání, editace, odstranění)
- Evidence zařízení podle sériových čísel
- Správu lokací a pozic, kde může být zařízení umístěno
- Přiřazování zařízení k jednotlivým místům
- Přehled a reporting všech zařízení

## Jak?
- Primárně jako webová aplikace dostupná z každého zařízení prostřednictvím webového prohlížeče
- Frontend: React framework pro moderní uživatelské rozhraní
- Backend: ASP.NET Core pro serverovou část aplikace
- Databáze: MariaDB pro ukládání dat
- API: REST API pro komunikaci mezi frontendem a backendem
- Autentifikace: Cookies a JSON Web Token (JWT) pro zabezpečení přístupu

## Kdo?
**Uživatelé systému:**
- Administrátoři - kompletní správa systému a uživatelů
- Správci zařízení - přidávání, editace a správa zařízení
- Operátoři - základní prohlížení a reporting
- Technici - terénní pracovníci pro fyzické umístění zařízení (budoucí mobilní přístup)

## Kde?
Aplikace bude dostupná pouze v intranetu zákazníka a nebude přístupná z vnější sítě. Díky responzivnímu designu bude použitelná i na mobilních zařízeních.

## Kdy?
Systém bude spolehlivý a poběží 24/7 v Docker Compose kontejnerech s možností vysoké dostupnosti. Plánované údržby budou prováděny během nočních hodin s minimálním dopadem na uživatele. Systém bude pravidelně zálohován, aby nedošlo ke ztrátě dat. Architektura bude škálovatelná pro obsloužení většího počtu současných uživatelů.