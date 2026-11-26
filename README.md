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

**The Odds API no ofrece bet365 en España.** Se usa solo para el consenso del mercado: Pinnacle, Betfair y unas 15 casas más. La cuota real de bet365 sale de football-data en fútbol y de **OddsPapi** en el resto. El plan gratis de OddsPapi trae 250 consultas al mes, así que el radar solo las gasta en confirmar los mejores candidatos de las próximas 47 h: una consulta por deporte para la lista de partidos (cada 12 h) y una por partido que trae **bet365 y Pinnacle a la vez**, con la misma hora.

## La web

Terminal oscura de análisis (Streamlit), pensada para PC y móvil:
- **Radar de la jornada:** todos los partidos analizados con su selección más interesante, probabilidad real frente a la que implica la cuota de bet365, valor esperado, semáforo (verde, ámbar, rojo, gris) y recomendación. Nunca queda vacía aunque no haya avisos.
- **Jugadas:** Élite, Sólidas, Por confirmar y Revisar, con barra de probabilidad, cuota justa y stake.
- **Combinadas:** solo con élite y sólidas de partidos y ligas distintas; probabilidad real conjunta frente a la cuota total.
- **Partidos:** fichas por competición con todos los mercados y clasificaciones (ranking Elo por superficie en tenis).
- **Asistente de apuesta:** presupuesto, fecha límite, riesgo y tipo (simple, combinada o repartir).
- **Rendimiento:** curva de banca y yield (Plotly), caída máxima, CLV, desglose por deporte, mercado, rango de cuota, competición y nivel, y calibración. Respeta `METRICS_SINCE`.
- **Estado y método:** fuentes en verde o rojo, consultas gastadas, pesos y backtest.

Los datos se leen en `radar/webdata.py` (funciones probadas) y se cachean con `@st.cache_data` por fecha del fichero de la base de datos: la web carga al instante y se refresca sola tras cada análisis.

## Cómo decide

1. Estima la probabilidad real de cada resultado con el mercado sin margen, usando **solo casas válidas**: mercado completo, margen creíble (máx. 1,12; exchanges 0,99-1,04) y cuotas tomadas como mucho 6 h antes o después de la de bet365. Si hay casas sharp (Pinnacle, exchanges) se usan solo ellas, y bet365 nunca entra en su propio consenso. El modelo propio está congelado (peso 0) en fútbol y tenis hasta que el backtest demuestre que mejora al mercado.
2. Compara esa probabilidad con la cuota de bet365: `valor esperado = probabilidad × cuota − 1`.
3. Clasifica cada jugada:

| Nivel | Qué significa | Dónde sale |
|---|---|---|
| 💎 **Élite** | Cuota de bet365 confirmada (menos de 24 h) con valor ≥ +2,5 % sobre el precio justo limpio, cuota 1,30-2,60. Si además hay referencia sharp (Pinnacle o exchange) y el valor es ≥ +3 %, va a Telegram (máx. 5 al día) | Web y Telegram |
| 🟢 **Sólida** | Probabilidad real ≥ 65 % y bet365 paga al menos el precio justo (valor ≥ −0,5 %), cuota 1,30-2,60. No es ventaja: es no regalar margen. Base de las combinadas | Web |
| **Por confirmar** | El mercado sugiere valor, pero falta la cuota de bet365 o es vieja. Se indica la cuota mínima | Web |
| **Revisar** | Valor > +15 %: casi siempre es un error de datos. Apuesta 0 | Web |
| **Sin ventaja** | Lo más probable del partido aunque bet365 cobre margen. Información, no recomendación | Radar |

- **Riesgo:** bajo (prob. ≥ 60 %), medio (40-60 %) o alto (< 40 %).
- **Seguimiento honesto:** una jugada que sale como «con valor» queda congelada con la cuota, la probabilidad y el stake de ese momento; no se reescribe ni se borra. El CLV solo se mide contra el cierre de Pinnacle.
- **Combinadas:** conservadora, equilibrada y arriesgada, siempre en bet365, solo con élite y sólidas, una por partido y una por liga. «Con valor» si la combinada supera el +2 % y todas sus patas tienen valor ≥ 0; «Sólida» si no baja del −1,5 %.

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

**Medir el rendimiento desde una fecha (`METRICS_SINCE`).** Para que la web y el resumen semanal cuenten solo las jugadas del motor nuevo:
1. GitHub → *Settings → Secrets and variables → Actions → Variables → New repository variable*: nombre `METRICS_SINCE`, valor `AAAA-MM-DD` (p. ej. `2026-11-27`).
2. Streamlit Community Cloud → tu app → *Settings → Secrets*: añade la línea `METRICS_SINCE = "2026-11-27"` y guarda (la app se reinicia sola).
3. En la web, *Rendimiento* permite cambiar a «Todo el historial» o elegir otra fecha sin tocar la configuración.

**Variables opcionales** (pestaña *Variables*):
- `SPORTS`: por ejemplo `football,tennis` para seguir solo esos deportes.
- `BANKROLL`, `MIN_EV`, `ALERT_EV` y `ALERT_MAX_DAY`.
- `MIN_ODDS` / `MAX_ODDS` (1,30 / 2,60), `SOLID_MIN_PROB` (0,65) y `SOLID_MIN_EV` (−0,005).

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
  combos.py                  combinadas descorrelacionadas (élite y sólidas)
  webdata.py                 datos de la web (rendimiento, radar del día)
  learn.py                   liquidar, métricas, recalibrar
  pipeline.py                orquestación
  notify.py                  Telegram
app.py                       web
```
