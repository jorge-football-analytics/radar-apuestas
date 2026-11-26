from .graph_objects import Figure
def make_subplots(rows=1, cols=1, specs=None, **k):
    return Figure(rows=rows, specs=specs)
