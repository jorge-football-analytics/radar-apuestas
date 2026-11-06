# 📡 Radar de apuestas · bet365

Compara las cuotas de **bet365** con la probabilidad real de cada resultado en fútbol, tenis, baloncesto y NFL.
Te avisa por Telegram solo de las mejores jugadas y enseña todo lo demás, ordenado, en la web.

Coste: **0 €**. GitHub Actions lo ejecuta cada 3 horas y Streamlit Community Cloud sirve la web.

## Qué cubre

| Deporte | Competiciones | Modelo propio | Cuota de bet365 |
|---|---|---|---|
| ⚽ Fútbol | 37 ligas: España 1 y 2, Inglaterra 1 a 4, Italia, Alemania y Francia 1 y 2, Escocia 1 y 2, Países Bajos, Portugal, Bélgica, Turquía, Grecia, Argentina, Brasil, MLS, México, Japón, Austria, Suiza, Dinamarca, Polonia, Noruega, Suecia, Irlanda, Rumanía, China y Finlandia | Dixon-Coles (goles) | football-data.co.uk (gratis, martes y viernes) |
| ⚽ Copas UEFA | Champions, Europa League, Conference | No (solo mercado) | OddsPapi |
| 🎾 Tenis | Todos los torneos ATP y WTA activos | Elo general + por superficie | OddsPapi |
| 🏀 Baloncesto | NBA (con modelo), Euroliga (solo mercado) | Elo + puntos esperados | OddsPapi · partidos y líneas de referencia gratis de ESPN |
| 🏈 NFL | Temporada regular y playoffs | Elo + puntos esperados | OddsPapi · partidos y líneas de referencia gratis de ESPN |

**The Odds API no ofrece bet365 en España.** Se usa solo para el consenso del mercado: Pinnacle, Betfair y unas 15 casas más. La cuota real de bet365 sale de football-data en fútbol y de **OddsPapi** en el resto. El plan gratis de OddsPapi trae 250 consultas al mes, así que el radar solo las gasta en confirmar los mejores candidatos.

## La web

- **Inicio:** las mejores jugadas, partidos por deporte y lo que empieza pronto.
- **Asistente de apuesta:** le dices presupuesto, fecha y hora límite, riesgo y tipo (simple, combinada o repartir). Elige la mejor opción entre todas las selecciones con cuota de bet365 que terminan antes.
- **Fútbol, Tenis, Baloncesto y NFL:** filtro por competición y riesgo, con tres pestañas:
  - **Jugadas:** con valor, por confirmar y pronósticos.
  - **Partidos:** probabilidades, cuotas de bet365 con su movimiento, cuota justa, goles o puntos esperados, Elo, cara a cara y todos los mercados.
  - **Clasificación:** forma y fuerza del modelo; ranking Elo por superficie en tenis.
- **Combinadas**, **Rendimiento** (curva, por competición y calibración) y **Estado y método** (cada fuente de datos en verde o rojo).

## Cómo decide

1. Estima la probabilidad real de cada resultado mezclando su modelo con el consenso del mercado sin margen (Pinnacle pesa doble). El peso del modelo lo fija el backtest y se reajusta con cada partido jugado.
2. Compara esa probabilidad con la cuota de bet365: `valor esperado = probabilidad × cuota − 1`.
3. Clasifica cada jugada:

| Nivel | Qué significa | Dónde sale |
|---|---|---|
| 💎 **Élite** | Valor ≥ +5 %, modelo y mercado de acuerdo, cuota entre 1,30 y 4 | Telegram (máx. 5 al día) y la web |
| **Con valor** | Cuota de bet365 confirmada con valor ≥ +3 % | Web |
| **Por confirmar** | El mercado sugiere valor, pero falta la cuota de bet365. Se indica la cuota mínima | Web |
| **Pronóstico** | La mejor opción del partido aunque no tenga ventaja | Web |

- **Riesgo:** bajo (prob. ≥ 60 %), medio (40-60 %) o alto (< 40 %).
- **Combinadas:** conservadora, equilibrada y arriesgada, siempre en bet365.

En Telegram recibes:
- las jugadas élite;
- como mucho una combinada con valor al día;
- un resumen cada mañana;
- un balance los lunes.

## Lo que dicen tus datos (léelo)

El backtest con datos reales del último año (13.000 selecciones de las 5 grandes ligas) dice dos cosas:
- **El modelo de goles no mejora al mercado.** Por eso su peso ha bajado a 0 %, y el radar confía en el consenso de casas.
- **bet365 casi nunca paga por encima del precio justo** en las grandes ligas: solo en un ~1 % de las selecciones, y por poco.

Por eso:
- Habrá **pocas jugadas con valor** y muchos días sin avisos. Es lo esperado: el filtro está para eso.
- Las ligas secundarias, el tenis y la NBA tienen más margen de error de bet365. Por eso se han añadido.
- Mira la **ventaja frente al cierre** en la web. Si es positiva tras 100 o más jugadas, la ventaja es real.
- **Pronósticos y combinadas de pronósticos no tienen ventaja.** Son informativos; apuesta poco o nada.
- Juega solo con dinero que puedas perder. Si deja de ser divertido: jugarbien.es o la autoexclusión (RGIAJ).

## Puesta en marcha

### Claves (Secrets de GitHub)
`Settings → Secrets and variables → Actions → New repository secret`:

| Secret | Para qué | Dónde |
|---|---|---|
| `ODDS_API_KEY` | Consenso del mercado | the-odds-api.com (gratis) |
| `ODDSPAPI_KEY` | Cuotas de bet365 de tenis, NBA, NFL y copas | oddspapi.io (gratis) |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Avisos | @BotFather |

**Variables opcionales** (pestaña *Variables*):
- `SPORTS`: por ejemplo `football,tennis` para seguir solo esos deportes.
- `BANKROLL`, `MIN_EV`, `ALERT_EV` y `ALERT_MAX_DAY`.

### En local
```bash
pip install -r requirements.txt
python -m radar run          # análisis completo
python -m radar backtest     # backtest de fútbol y tenis
streamlit run app.py         # la web
python -m pytest             # pruebas
```

## Estructura
```
radar/
  config.py                  deportes, ligas y ajustes
  sources/football_data.py   fútbol: histórico + cuotas de bet365
  sources/tennis_data.py     tenis: histórico con superficie y cuotas
  sources/espn.py            NBA y NFL: resultados
  sources/odds_api.py        consenso del mercado
  sources/oddspapi.py        cuotas de bet365 (tenis, NBA, NFL, copas)
  model/dixon_coles.py       fútbol
  model/elo.py               tenis, NBA, NFL
  value.py                   valor, riesgo, filtro élite
  combos.py                  combinadas por riesgo
  learn.py                   liquidar, métricas, recalibrar
  pipeline.py                orquestación
  notify.py                  Telegram
app.py                       web
```
