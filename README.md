# 📡 Radar de apuestas · bet365

Compara las cuotas de **bet365** con la probabilidad real de cada resultado en fútbol, tenis, baloncesto y NFL.
Te avisa por Telegram solo de las mejores jugadas y enseña todo lo demás, ordenado, en la web.

Coste: **0 €**. GitHub Actions lo ejecuta cada 3 horas y Streamlit Community Cloud sirve la web.

## Qué cubre

| Deporte | Competiciones | Datos y resultados | Cuota de bet365 |
|---|---|---|---|
| ⚽ Fútbol | 37 ligas: España 1 y 2, Inglaterra 1 a 4, Italia, Alemania y Francia 1 y 2, Escocia 1 y 2, Países Bajos, Portugal, Bélgica, Turquía, Grecia, Argentina, Brasil, MLS, México, Japón, Austria, Suiza, Dinamarca, Polonia, Noruega, Suecia, Irlanda, Rumanía, China y Finlandia | football-data.co.uk (histórico, Dixon-Coles) | football-data (martes y viernes) y OddsPapi en bloque |
| ⚽ Copas UEFA | Champions, Europa League, Conference | ESPN (resultados para liquidar, gratis) | OddsPapi en bloque |
| 🎾 Tenis | ATP y WTA activos | ESPN (Elo general y por superficie). tennis-data.co.uk bloquea las IP de GitHub | OddsPapi en bloque |
| 🏀 Baloncesto | NBA (con modelo), Euroliga (solo mercado) | ESPN día a día (Elo + puntos esperados) | OddsPapi en bloque · líneas de DraftKings gratis de ESPN |
| 🏈 NFL | Temporada regular y playoffs | ESPN semana a semana (Elo) | OddsPapi en bloque · líneas de DraftKings gratis de ESPN |

**Mercados por partido:** ganador (1X2 o moneyline), doble oportunidad, más/menos (goles o puntos), ambos marcan y hándicap, cada uno con su probabilidad real, cuota justa, cuota de bet365 y valor.

**Cuota operable (v6).** Si bet365 no tiene cuota reciente, el radar no se bloquea: usa la **mejor cuota real** de una casa generalista (William Hill, Unibet, Betsson, 888sport, Marathonbet, DraftKings…), dice cuál es y estima lo que pagaría bet365 (`Cuota ref. mercado 1,85 · William Hill · bet365 ~1,83`), además de la cuota mínima a la que cogerla en bet365. Casas offshore y exchanges nunca sirven de referencia operable, y si la mejor cuota supera en más de un 4 % a la segunda se usa la segunda. Telegram sigue avisando solo con bet365 confirmada.

**The Odds API no ofrece bet365 en España.** Se usa para el consenso del mercado y las cuotas de referencia: Pinnacle, Betfair y unas 15 casas más, ahora también en las ligas nacionales de fútbol y con hándicap y totales en NBA y NFL. El presupuesto se reparte solo: cada 6 h las ligas que juegan en las próximas 24 h, una vez al día el resto. La cuota real de bet365 sale de football-data en fútbol y de **OddsPapi** en el resto. El plan gratis de OddsPapi trae 250 consultas al mes, así que el radar solo las gasta en confirmar los mejores candidatos de las próximas 47 h: una consulta por deporte para la lista de partidos (cada 12 h) y una por partido que trae **bet365 y Pinnacle a la vez**, con la misma hora.

## La web

Plataforma oscura de análisis (Streamlit), pensada para PC y móvil:
- **Centro de mando:** las 4 mejores oportunidades del día en tarjetas (escudos, cuota y casa, velocímetro de confianza, probabilidad real frente a mercado y análisis táctico desplegable), la combinada recomendada del día, el resto de recomendaciones (8-15 al día) y la tabla interactiva de toda la jornada con filtros.
- **Jugadas:** los 4 niveles (💎 Élite, 🛡️ Sólidas, ⚖️ Valor moderado, 🔥 Cuota alta) más Por confirmar y Revisar, con velocímetro de confianza (0-100), barra de probabilidad, cuota justa, stake y calculadora de stake/beneficio.
- **Combinadas:** tres perfiles cada jornada (Segura 1,70-2,20 · Equilibrada 2,50-4,00 · Valor EV+ cuando hay ventaja), de cualquier mercado, una pata por partido, con el porqué de cada selección y la probabilidad real conjunta frente a la cuota total.
- **Partidos:** ficha profesional de cada partido con gráficos Plotly (probabilidad real frente al mercado, marcadores más probables o distribución del margen, ataque y defensa, racha de los últimos 5), bajas y lesiones, análisis y todos los mercados con su mejor cuota. Clasificaciones y ranking Elo por superficie en tenis.
- **Asistente de apuesta (copiloto):** banca, riesgo, deportes, cuota objetivo y fecha límite. El riesgo fija el rango de cuota (bajo 1,40-1,90 · medio 1,90-2,80 · alto 2,80-5,00) y el deslizador lo afina; **el rango se respeta siempre**. Orden de preferencia: cuotas de bet365 confirmadas con valor ≥ 0 → selecciones «por confirmar» con la cuota mínima a la que cogerlas → si nada encaja, lo dice y enseña la alternativa más cercana (nunca una apuesta con valor negativo como opción principal). Devuelve siempre 3 propuestas jugables: Conservadora, Equilibrada y Alto retorno (dentro del rango siempre que lo analizado lo permita; si no, marcada como fuera de rango), con stake (Kelly fraccionado si hay ventaja; 0,5 % a precio justo; 0 si el valor es negativo) y el porqué de cada selección.
- **Rendimiento:** seguimiento en vivo (curva de banca y yield, caída máxima, CLV, desglose por deporte, mercado, cuota, competición, nivel y casa, calibración) y, mientras no haya apuestas resueltas, el backtest histórico con su intervalo de confianza. Respeta `METRICS_SINCE`.
- **Estado y método:** fuentes en verde o rojo, consultas gastadas, pesos y backtest.

Los datos se leen en `radar/webdata.py` (funciones probadas) y se cachean con `@st.cache_data` por fecha del fichero de la base de datos: la web carga al instante y se refresca sola tras cada análisis.

## Cómo decide

1. Estima la probabilidad real de cada resultado con el mercado sin margen, usando **solo casas válidas**: mercado completo, margen creíble (máx. 1,12; exchanges 0,99-1,04) y cuotas tomadas como mucho 6 h antes o después de la de bet365. Si hay casas sharp (Pinnacle, exchanges) se usan solo ellas, y bet365 nunca entra en su propio consenso. El modelo propio está congelado (peso 0) en fútbol y tenis hasta que el backtest demuestre que mejora al mercado.
2. Compara esa probabilidad con la cuota operable (bet365 si es reciente; si no, la mejor cuota real de otra casa, nombrada): `valor esperado = probabilidad × cuota − 1`.
3. Clasifica cada jugada:

| Nivel | Qué significa | Dónde sale |
|---|---|---|
| 💎 **Élite** | Cuota operable con valor ≥ +2,5 % sobre el precio justo limpio, cuota 1,30-2,60 (con cuota de otra casa, además referencia sharp). Con bet365 confirmada, referencia sharp y valor ≥ +3 %, va a Telegram (máx. 5 al día) | Web y Telegram |
| 🛡️ **Sólida** | Probabilidad real ≥ 65 % y la cuota paga al menos el precio justo (valor ≥ −0,5 %), cuota 1,20-2,60. No es ventaja: es no regalar margen. Base de las combinadas | Web |
| ⚖️ **Valor moderado** | Cuota confirmada con valor entre 0 y +2,5 % y probabilidad ≥ 40 %, cuota 1,30-2,60. Ventaja pequeña: stake pequeño | Web |
| 🔥 **Cuota alta** | Cuota 2,40-3,80 con valor ≥ +1 % y probabilidad ≥ 25 %. Más varianza: stake Kelly reducido | Web |
| 🎯 **Por confirmar / cuota objetivo** | No hay cuota operable (o paga menos que el precio justo): se indica la cuota mínima a la que cogerla en bet365 | Web |
| **Revisar** | Valor > +15 %: casi siempre es un error de datos. Apuesta 0 | Web |
| **Sin ventaja** | Lo más probable del partido aunque bet365 cobre margen. Información, no recomendación | Radar |

- **Riesgo:** bajo (prob. ≥ 60 %), medio (40-60 %) o alto (< 40 %).
- **Seguimiento honesto:** una jugada que sale como «con valor» queda congelada con la cuota, la probabilidad y el stake de ese momento; no se reescribe ni se borra. El CLV solo se mide contra el cierre de Pinnacle.
- **Combinadas:** Segura, Equilibrada y Valor EV+, siempre en bet365, una pata por partido y, si se puede, una por liga. «Con valor» si la combinada supera el +2 % y todas sus patas tienen valor ≥ 0; «Sólida» si no baja del −1,5 %; «Sin ventaja» si bet365 se queda más margen (se indica); «Por confirmar» si alguna pata aún no tiene cuota de bet365.

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
- `MODERATE_MIN_PROB` (0,40), `HIGH_ODDS_MIN` / `HIGH_ODDS_MAX` (2,40 / 3,80) y `HIGH_MIN_EV` (0,01).
- `SOLID_MIN_ODDS` (1,20), `REF_MAX_AGE_H` (12: antigüedad máxima de la cuota de referencia), `ODDS_LOOKAHEAD_H` (72),
  `ODDS_API_FOOTBALL` (1) y `ODDS_MARKETS_NBA` / `ODDS_MARKETS_NFL` (`h2h,spreads,totals`).

### Bajas y lesiones
El informe de cada partido muestra las bajas confirmadas: **API-Football** en fútbol (una consulta por día de partidos, con caché de 6 h; el plan gratis da 100 al día) y el parte de lesiones de **ESPN** en NBA y NFL. Para activarlo en fútbol añade el secreto `APIFOOTBALL_KEY` en GitHub → *Settings → Secrets and variables → Actions*. Si falta la clave, o tu plan no cubre la temporada actual, el informe lo dice («sin datos para este partido») en vez de inventar.

### Si Streamlit Cloud muestra `AttributeError: module 'radar...' has no attribute ...`
Tras un push, Streamlit vuelve a ejecutar `app.py` pero conserva en memoria los módulos de `radar/` que ya había importado
(con el código antiguo). `app.py` ya los recarga solo cuando cambian; si alguna vez vuelve a pasar: *Manage app → ⋮ →
Reboot app*.

### Comprobar las 7 páginas sin servidor
```bash
python tools/check_pages.py data/radar.db          # OK/FALLO por página (errores y textos rotos nan/None)
python tools/check_assistant.py data/radar.db      # 162 combinaciones de riesgo/cuota/tipo/fecha: el asistente respeta el filtro
STUB_OVERRIDES='{"m_sp":"basketball","pf_tier":"combinadas"}' python tools/check_pages.py
```

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
  combos.py                  combinadas por perfil (segura, equilibrada, valor)
  stats.py                   forma, racha, local/visitante y cara a cara
  model/derived.py           mercados derivados (doble oportunidad, más/menos, ambos marcan, hándicap)
  webdata.py                 datos de la web (rendimiento, radar del día)
  learn.py                   liquidar, métricas, recalibrar
  pipeline.py                orquestación
  notify.py                  Telegram
app.py                       web
```
