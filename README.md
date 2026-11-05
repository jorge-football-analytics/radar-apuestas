# ⚽ Radar de apuestas

Analiza automáticamente los partidos de LaLiga, Premier, Serie A, Bundesliga, Ligue 1 y Champions. Compara la probabilidad real estimada con las cuotas de bet365 y del resto del mercado, y te avisa por Telegram cuando encuentra una **apuesta de valor** o una **combinada** con valor esperado positivo. Cada partido jugado se usa para medir el sistema y recalibrarlo.

Coste: **0 €**. GitHub Actions ejecuta el análisis y Streamlit Community Cloud sirve la web.

---

## Qué hace en cada ejecución (cada 3 horas)

1. **Descarga datos** de football-data.co.uk: 4 temporadas de resultados de cada liga, con cuotas de apertura y cierre de bet365 y Pinnacle, más las cuotas de la próxima jornada.
2. **Ajusta un modelo Dixon-Coles por liga.** Estima la fuerza de ataque y defensa de cada equipo, la ventaja de jugar en casa y la corrección de marcadores bajos. Los partidos recientes pesan más (vida media de un año aproximadamente).
3. **Descarga cuotas en vivo** de ~40 casas, bet365 incluida, con The Odds API. Solo gasta créditos en las ligas con partidos en las próximas 72 h y reparte los 500 créditos del mes para no quedarse sin ellos.
4. **Bajas:** lesionados y sancionados de API-Football (si tu plan lo permite) y las que apuntes en `data/bajas_manual.csv`. Rebajan los goles esperados del equipo.
5. **Calcula la probabilidad final** como mezcla del modelo y del consenso del mercado sin margen. El peso de cada parte lo decide el backtest y luego los resultados reales.
6. **Detecta valor** con `EV = probabilidad × cuota − 1`. Busca la mejor cuota del mercado y la de bet365, y sugiere un stake con Kelly fraccional.
7. **Monta combinadas** conservadoras, equilibradas y arriesgadas. Todas las patas van en la misma casa y cada una tiene valor esperado positivo.
8. **Avisa por Telegram** de lo nuevo con EV ≥ +6 % y confianza media o alta. Nunca repite un aviso. Cada lunes manda un resumen de rendimiento.
9. **Aprende:** liquida los partidos jugados y mide acierto, ROI, CLV y calidad de las probabilidades. Con esos datos recalibra el peso modelo/mercado.

### Qué significa la confianza

| Confianza | Criterio |
|---|---|
| 🟢 **Alta** | El modelo propio y el consenso del mercado coinciden en que la cuota está por encima de lo justo, con 4 casas o más. |
| 🟡 **Media** | Solo el modelo ve valor, o no hay modelo pero la cuota supera claramente el consenso de 6 casas o más. |
| 🟠 **Baja** | El valor sale solo de la mezcla. Se guarda y se mide, pero no se avisa. |

---

## Qué esperar (léelo antes de apostar)

- **No existe "alto acierto + cuota alta".** Una cuota de 3.00 acierta como mucho un tercio de las veces. El objetivo es el **valor**: ganar dinero a largo plazo aunque se pierdan muchas apuestas.
- **El mercado es muy eficiente** en las grandes ligas. La ventaja suele venir de encontrar casas que pagan más que el consenso, más que de que el modelo "sepa más" que todos.
- **La métrica que manda es el CLV** (*closing line value*). Si sistemáticamente apuestas a mejor cuota que la de cierre, tienes ventaja real. Con 50 o 100 apuestas, el ROI es casi todo suerte.
- **Mira el backtest** en la pestaña 🧪. Si el intervalo del ROI incluye el 0, todavía no hay evidencia de ventaja. Lo sensato es empezar con stakes mínimos o solo observando.
- **bet365 limita las cuentas que ganan** de forma constante. Ten cuenta en varias casas.
- **Las cuotas de football-data pueden tener 1-2 días.** El aviso lo indica. Comprueba siempre la cuota actual antes de apostar.
- **Juega solo con una banca que puedas perder.** Si deja de ser divertido: jugarbien.es o el registro de autoexclusión (RGIAJ).

---

## Puesta en marcha

### 1. Claves gratuitas (10 minutos)

| Servicio | Para qué | Dónde |
|---|---|---|
| The Odds API | Cuotas de bet365 y ~40 casas | https://the-odds-api.com → plan Starter |
| Telegram | Avisos al móvil | Habla con **@BotFather** → `/newbot` → copia el token |
| API-Football (opcional) | Lesiones y sanciones | https://dashboard.api-football.com. El plan gratis no incluye la temporada actual; el Pro sí, por unos 19 $/mes |

### 2. Probar en tu PC

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (en Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env           # y rellena las claves
```

Escribe cualquier cosa a tu bot en Telegram y después:

```bash
python -m radar telegram-test    # muestra tu chat_id → ponlo en TELEGRAM_CHAT_ID y repite
python -m radar backtest         # ¿habría ganado el último año? (1-2 min)
python -m radar run              # primer análisis completo
streamlit run app.py             # la web en http://localhost:8501
```

### 3. Dejarlo funcionando solo en internet (gratis)

1. Crea un repositorio **privado** en GitHub y sube esta carpeta.
2. En el repo, ve a **Settings → Secrets and variables → Actions → New repository secret** y crea `ODDS_API_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` y, si lo tienes, `APIFOOTBALL_KEY`.
   En la pestaña **Variables** puedes ajustar `BANKROLL`, `MIN_EV` y `ALERT_EV`.
3. Ve a **Actions → Radar de apuestas → Run workflow** para la primera ejecución. A partir de ahí corre solo cada 3 horas y guarda la base de datos en el repo.
4. **Web:** entra en https://share.streamlit.io con tu GitHub, haz **New app**, elige el repo y pon `app.py` como fichero principal. En los ajustes de la app, limita quién puede verla a tu email.
   La web se actualiza sola con cada análisis.

Uso de GitHub Actions: unos 2 minutos por ejecución × 8 al día ≈ 500 min/mes, dentro de los 2.000 gratis de un repo privado.

---

## Ajustes útiles (`.env` o Variables de GitHub)

| Variable | Por defecto | Qué hace |
|---|---|---|
| `BANKROLL` | 100 | Banca en €. Los stakes son un % de ella. |
| `KELLY_FRACTION` | 0.25 | Agresividad del stake. 0.25 de Kelly es prudente. |
| `MAX_STAKE_PCT` | 0.03 | Tope por apuesta: 3 % de la banca. |
| `MIN_EV` / `ALERT_EV` | 0.04 / 0.06 | Valor mínimo para guardar / para avisar. |
| `MIN_PROB` / `MAX_ODDS` | 0.25 / 6.0 | Evita marcar *longshots* poco fiables. |
| `BET_BOOKS` | todas menos Pinnacle | Casas donde tienes cuenta. Con `bet365` todas las jugadas y combinadas serán en bet365; el resto de casas siguen sirviendo para calcular el precio justo. |
| `ODDS_BOOKMAKERS` | — | Hasta 10 casas concretas por el coste de una región. |

## Bajas a mano

Edita `data/bajas_manual.csv`:

```csv
liga,equipo,jugador,importancia,hasta
laliga,Real Madrid,Mbappé,3,2026-11-20
premier,Arsenal,Saka,2,2026-11-15
```

- **Ligas:** laliga, premier, seriea, bundesliga, ligue1, champions.
- **Equipo:** con el nombre que usa football-data (Ath Madrid, Man United, Paris SG…).
- **Importancia:** 1 = rotación, 2 = titular, 3 = figura. Cada punto resta un 2 % de los goles esperados, con un máximo del 12 %.

## Cuando quieras invertir

En orden de impacto:

1. **The Odds API de 30 $/mes:** cuotas casi en tiempo real, más mercados (ambos marcan, hándicap) y más ligas.
2. **API-Football Pro, unos 19 $/mes:** lesiones, sanciones y alineaciones de la temporada actual. Se activa solo con poner la clave.
3. Añadir xG (expected goals) al modelo y otros deportes (tenis con Elo, NBA).

## Estructura

```
radar/
  sources/football_data.py   histórico, cuotas de cierre, próxima jornada
  sources/odds_api.py        cuotas en vivo + presupuesto de créditos
  sources/injuries.py        bajas (API-Football + CSV manual)
  model/dixon_coles.py       modelo de goles
  model/markets.py           1X2, más/menos, ambos marcan, quitar margen
  value.py                   detector de valor + stake
  combos.py                  combinadas por riesgo
  learn.py                   liquidar, métricas, CLV, recalibrar
  backtest.py                walk-forward
  pipeline.py                orquestación
  notify.py                  Telegram
app.py                       web (Streamlit)
.github/workflows/radar.yml  ejecución automática cada 3 h
tests/                       pruebas con una liga sintética (pytest)
```
