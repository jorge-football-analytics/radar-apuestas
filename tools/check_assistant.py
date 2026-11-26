"""Barrido del asistente sobre una base de datos real: 162 combinaciones de riesgo, rango de cuota, tipo y fecha límite.
Comprueba que la opción principal respeta siempre el rango pedido con valor ≥ 0 o, si nada encaja, se marca como
«alternativa más cercana».  Uso:  python tools/check_assistant.py data/radar.db   (código 1 si hay fallos)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datetime import datetime, timedelta, timezone
from radar import webdata, assistant
db = sys.argv[1] if len(sys.argv) > 1 else "data/radar.db"
pool = webdata.load_dashboard(db)["pool"]; now = datetime.now(timezone.utc)
bad=tot=plans=0
for risk in ("bajo","medio","alto"):
    for rng in (None,(2.0,3.4),(1.4,1.7),(3.0,6.0),(1.2,8.0),(1.5,1.6)):
        for tipo in ("auto","simple","combinada"):
            for days in (1,3,7):
                plan,alts,n = assistant.copilot(pool,100,now+timedelta(days=days),now,risk,tipo,None,rng); tot+=1
                if plan is None: continue
                plans+=1
                lo,hi = rng or assistant.RISK_RANGE[risk]; inr = lo-1e-9<=plan.odds<=hi+1e-9
                ok = (plan.status in ("exacta","por_confirmar") and inr and plan.ev>=0) or plan.status=="cercana"
                ok = ok and not (plan.status=="cercana" and inr and plan.ev>=0)
                ok = ok and all(a.role in ("conservadora","retorno") for a in alts)
                if not ok: bad+=1; print("FALLO", risk,rng,tipo,days,plan.odds,plan.ev,plan.status)
print(f"{db}: {tot} combinaciones · {plans} con plan · {bad} fallos")
sys.exit(1 if bad else 0)
