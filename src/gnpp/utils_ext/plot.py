colors = {
    'b': 'blue',
    'g': 'green',
    'r': 'red',
    'c': 'cyan',
    'm': 'magenta',
    'y': 'yellow',
    'k': 'black',
    # 'w': 'white',
}
line_styles = {
    # '-': 'solid line style',
    '--': 'dashed line style',
    '-.': 'dash-dot line style',
    # ':': 'dotted line style',
}
markers = {
    # '.': 'point marker',
    # ',': 'pixel marker',
    'o': 'circle marker',
    'v': 'triangle_down marker',
    '^': 'triangle_up marker',
    '<': 'triangle_left marker',
    '>': 'triangle_right marker',
    '1': 'tri_down marker',
    '2': 'tri_up marker',
    '3': 'tri_left marker',
    '4': 'tri_right marker',
    '8': 'octagon marker',
    's': 'square marker',
    'p': 'pentagon marker',
    'P': 'plus (filled) marker',
    '*': 'star marker',
    'h': 'hexagon1 marker',
    'H': 'hexagon2 marker',
    '+': 'plus marker',
    'x': 'x marker',
    'X': 'x (filled) marker',
    'D': 'diamond marker',
    # 'd': 'thin_diamond marker',
    # '|': 'vline marker',
    # '_': 'hline marker',
}

def fmt_iterator(colors, line_styles, markers):
    from itertools import cycle
    colors_iter = cycle(colors)
    line_styles_iter = cycle(line_styles)
    markers_iter = cycle(markers)
    while True:
        fmt = next(colors_iter)+next(line_styles_iter)+next(markers_iter)
        yield fmt