"""Sustituto mínimo de Streamlit para tools/check_pages.py: ejecuta todas las páginas de app.py sin servidor y
guarda el HTML que pintarían. No sustituye a Streamlit en producción."""
import os
import traceback
OUT = {"page": "_", "html": {}, "calls": [], "submit": os.getenv("STUB_SUBMIT", "1") == "1",
       "overrides": {}, "errors": {}}


class StopError(Exception):
    pass


def _emit(kind, txt):
    OUT["html"].setdefault(OUT["page"], []).append((kind, txt))


def set_page_config(**k): pass
def markdown(txt, unsafe_allow_html=False): _emit("md", txt)
def caption(txt): _emit("cap", f'<p class="hint">{txt}</p>')
def plotly_chart(fig, **k):
    fig.validate()
    _emit("chart", '<div class="empty">[gráfico plotly: %d trazas]</div>' % len(fig.traces))
def dataframe(df, column_config=None, **k):
    if column_config:
        cols = set(getattr(getattr(df, "data", df), "columns", []))
        missing = [c for c in column_config if c not in cols]
        if missing:
            raise KeyError(f"column_config con columnas inexistentes: {missing}")
    data = getattr(df, "data", df)  # Styler → DataFrame
    if hasattr(df, "to_html"):
        try:
            df.to_html()
        except Exception as ex:
            raise RuntimeError(f"Styler falla: {ex}")
    _emit("df", data.to_html(classes="tb") if hasattr(data, "to_html") else str(data))
def stop(): raise StopError()


class _ColumnConfig:
    """st.column_config.*: devuelve un dict con lo que se pidió (solo se comprueba que exista y se pueda llamar)."""
    def __getattr__(self, name):
        if name not in ("TextColumn", "NumberColumn", "ProgressColumn", "LinkColumn", "DatetimeColumn",
                        "CheckboxColumn", "ListColumn", "LineChartColumn", "BarChartColumn", "Column", "ImageColumn"):
            raise AttributeError(name)
        return lambda *a, **k: {"tipo": name, **k}


column_config = _ColumnConfig()
def container(**k): return _Ctx()
def divider(): _emit("md", "<hr>")


def cache_data(*a, **k):
    if a and callable(a[0]):
        return a[0]
    return lambda f: f


def _fmt(options, format_func):
    if format_func:
        for o in options:
            format_func(o)


session_state: dict = {}


def _ov(key, default):
    if key in OUT["overrides"]:
        return OUT["overrides"][key]
    if key is not None and key in session_state:
        return session_state[key]
    if key is not None and default is not None:
        session_state[key] = default
    return default


def pills(label, options, default=None, format_func=None, key=None, **k):
    _fmt(options, format_func); return _ov(key, default)
def segmented_control(label, options, default=None, format_func=None, key=None, selection_mode="single", **k):
    _fmt(options, format_func); return _ov(key, default)
def selectbox(label, options, format_func=None, key=None, **k):
    options = list(options); _fmt(options, format_func); return _ov(key, options[0] if options else None)
def multiselect(label, options, default=None, format_func=None, key=None, **k):
    _fmt(list(options), format_func); return _ov(key, default)
def text_input(label, key=None, **k): return _ov(key, "")
def number_input(label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
    return _ov(key, value if value is not None else min_value)
def date_input(label, value=None, key=None, **k): return _ov(key, value)
def time_input(label, value=None, key=None, **k): return _ov(key, value)
def toggle(label, value=False, key=None, **k): return _ov(key, value)
def expander(label, expanded=False): return _Ctx()
def form_submit_button(*a, **k): return OUT["submit"]
def button(*a, **k): return OUT["submit"]


class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return globals()[name]


def columns(spec, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Ctx() for _ in range(n)]
def tabs(names):
    for n in names:
        _emit("tab", f'<div class="sec"><h3>[pestaña] {n}</h3></div>')
    return [_Ctx() for _ in names]
def form(key): return _Ctx()


class Page:
    def __init__(self, fn, title="", icon="", default=False, url_path=None):
        self.fn, self.title = fn, title


class _Nav:
    def __init__(self, pages): self.pages = pages
    def run(self):
        for sec, ps in self.pages.items():
            for p in ps:
                OUT["page"] = p.title
                try:
                    p.fn()
                except StopError:
                    pass
                except Exception:  # noqa: BLE001  — se registra y se sigue con la siguiente página
                    OUT["errors"][p.title] = traceback.format_exc()


def navigation(pages, position="sidebar"): return _Nav(pages)
sidebar = _Ctx()


def slider(label, min_value=None, max_value=None, value=None, step=None, key=None, **k):
    return _ov(key, value)
