import math
class _Trace(dict):
    def __init__(self, **k):
        super().__init__(**k)
class Scatter(_Trace): pass
class Bar(_Trace): pass
class Heatmap(_Trace): pass
class Indicator(_Trace): pass
class Figure:
    def __init__(self, data=None, rows=1, specs=None):
        self.traces = [data] if isinstance(data, _Trace) else list(data or [])
        self.rows, self.specs = rows, specs
    def add_trace(self, t, row=None, col=None, secondary_y=None):
        if row is not None and row > self.rows:
            raise ValueError("fila inexistente")
        if secondary_y and not (self.specs and self.specs[(row or 1) - 1][0].get("secondary_y")):
            raise ValueError("secondary_y sin specs")
        self.traces.append(t); return self
    def update_layout(self, **k): return self
    def update_xaxes(self, **k): return self
    def update_yaxes(self, **k): return self
    def add_hline(self, **k): return self
    def add_vline(self, **k): return self
    def add_annotation(self, **k): return self
    def add_shape(self, **k): return self
    def update_traces(self, **k): return self
    def validate(self):
        for t in self.traces:
            for key in ("x", "y", "z", "text"):
                v = t.get(key)
                if v is not None:
                    list(v)
