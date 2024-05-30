import math
import time
from functools import partial
from operator import attrgetter
from typing import Callable, List
from typing_local import LearningRateFunction, MinimizeSympyFunction
from grad_desc import constant_rate, dichotomy_method, golden_ratio_method
import funcs as fs
from all import *
import dash
from dash import dcc
from dash import html
import numpy as np
from dash.dependencies import Input, Output
import dash_daq as daq
import plotly.graph_objects as go

from utils import create_gradient_colormap, get_color


class Settings:
    LearningRateStrategies: List[LearningRateFactoryFunction] = [
        constant_rate, dichotomy_method, golden_ratio_method
    ]
    MinimizeMethods: List[MinimizeSympyFunction] = methods


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        # график
        html.Div([
            dcc.Graph(id='graph', figure={'layout': go.Layout(
                height=1100
            )}),
            # фото кота
            html.Div([
                html.Img(
                    src='https://koshka.top/uploads/posts/2021-11/1637915005_54-koshka-top-p-chernii-kot-za-kompyuterom-66.jpg',
                    style={'width': '100%'}),
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        html.Div([
            html.Div([
                html.P('Описание', style={}),
                html.Div(id='result-text'),
            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            # функция
            html.Div([
                html.P('Функция:', style={'margin-top': '10px'}),
                dcc.Input(id='function', type='text', value='x**2+y**2'),
            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            # метод поиска минимума функции
            html.Div([
                html.P('Метод поиска минимума функции', style={'margin-top': '10px'}),
                dcc.Dropdown(
                    id='method',
                    options=(m_opts := list(map(attrgetter('__name__'), Settings.MinimizeMethods))),
                    value=m_opts[0]
                ),
            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            html.Div([
                html.Label('Max Steps:', style={'margin-top': '10px'}),
                daq.NumericInput(id='max-steps', min=10, max=10000, value=1000, size=120),
            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            # начальная точка
            html.Div([
                html.P('Начальная точка', style={'margin-top': '10px'}),
                html.Div([
                    html.Label('x:', style={'margin-right': '10px'}),
                    dcc.Input(id='start-x', type='number', value=1, step=0.000001),

                ], style={'display': 'inline-block', 'margin-right': '10px'}),

                html.Div([
                    html.Label('y:', style={'margin-right': '10px'}),
                    dcc.Input(id='start-y', type='number', value=1, step=0.000001),

                ], style={'display': 'inline-block'}),

            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            # настройки шагов
            html.Div([
                html.Div([
                    html.P('Learning rate strategy:'),
                    dcc.Dropdown(
                        id='learning-rate-strategy',
                        options=(lr_opts := list(map(attrgetter('__name__'), Settings.LearningRateStrategies))),
                        value=lr_opts[0]),
                ], style={}),

                html.Div([
                    html.Label('Learning rate:', style={'margin-right': '10px'}),
                    dcc.Input(id='learning-rate', type='number', value=0.01, step=0.0001),
                ], style={'margin-right': '10px', 'padding-top': '10px'}),

            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            # критерии останова
            html.Div([
                html.P('Stop by delta f:', style={'margin-top': '10px'}),
                daq.BooleanSwitch(id='deltaf', on=True, label='delta f(x, y) < delta_f_threshold'),
                html.Label('Stop by delta f:', style={'margin-top': '10px', 'margin-right': '10px'}),
                dcc.Input(id='deltaf-threshold', type='number', value=0.000001, step=0.000001),

            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            html.Div([
                html.Label('Stop by delta xy:', style={'margin-top': '10px', 'margin-right': '10px'}),
                daq.BooleanSwitch(id='deltaxy', on=True, label='||(Δxk, Δyk)|| < delta_xy_threshold'),
                html.Label('Delta xy threshold:', style={'margin-top': '10px', 'margin-right': '10px'}),
                dcc.Input(id='deltaxy-threshold', type='number', value=0.00001, step=0.00001),
            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),

            # ограничения
            html.Div([
                html.P('Ограничения', style={'margin-top': '10px'}),
                html.Div([
                    html.Label('x upper limit:', style={'margin-right': '10px'}),
                    dcc.Input(id='x-upper-limit', type='number', value=10, step=0.1),
                ], style={'margin-top': '10px'}),

                html.Div([
                    html.Label('x lower limit:', style={'margin-right': '10px'}),
                    dcc.Input(id='x-lower-limit', type='number', value=-10, step=0.1),
                ], style={'margin-top': '10px'}),

                html.Div([
                    html.Label('y upper limit:', style={'margin-right': '10px'}),
                    dcc.Input(id='y-upper-limit', type='number', value=10, step=0.1),
                ], style={'margin-top': '10px'}),

                html.Div([
                    html.Label('y lower limit:', style={'margin-right': '10px'}),
                    dcc.Input(id='y-lower-limit', type='number', value=-10, step=0.1),
                ], style={'margin-top': '10px'}),

                html.Div([
                    html.Label('z upper limit:', style={'margin-right': '10px'}),
                    dcc.Input(id='z-upper-limit', type='number', value=10, step=0.1),
                ], style={'margin-top': '10px'}),

                html.Div([
                    html.Label('z lower limit:', style={'margin-right': '10px'}),
                    dcc.Input(id='z-lower-limit', type='number', value=-10, step=0.1),
                ], style={'margin-top': '10px'}),

            ], style={'border': '1px solid black', 'margin-top': '10px', 'padding': '10px'}),
        ],
            style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}
        ),

    ], style={'font-size': '20px', 'fontFamily': 'Arial, sans-serif'})
])


@app.callback(
    Output('graph', 'figure'),
    Output('result-text', 'children'),
    [
        Input('function', 'value'),
        Input('method', 'value'),
        Input('start-x', 'value'),
        Input('start-y', 'value'),
        Input('learning-rate', 'value'),
        Input('learning-rate-strategy', 'value'),
        Input('max-steps', 'value'),
        Input('deltaf', 'on'),
        Input('deltaf-threshold', 'value'),
        Input('deltaxy', 'on'),
        Input('deltaxy-threshold', 'value'),
        Input('x-upper-limit', 'value'),
        Input('x-lower-limit', 'value'),
        Input('y-upper-limit', 'value'),
        Input('y-lower-limit', 'value'),
        Input('z-upper-limit', 'value'),
        Input('z-lower-limit', 'value'),
    ]
)
def update_graph(
    function,
    method_name,
    start_x,
    start_y,
    learning_rate,
    learning_rate_strategy,
    max_steps,
    delta_f: bool,
    delta_f_threshold: float,
    delta_xy: bool,
    delta_xy_threshold: float,
    x_upper_limit,
    x_lower_limit,
    y_upper_limit,
    y_lower_limit,
    z_upper_limit,
    z_lower_limit
):
    minimize_method: Optional[MinimizeSympyFunction] = None
    for method in Settings.MinimizeMethods:
        if method.__name__ == method_name:
            minimize_method = method
            break
    if minimize_method is None:
        raise ValueError(f'Unknown method: {method_name}')

    learning_rate_function: Optional[LearningRateFactoryFunction] = None
    for lr in Settings.LearningRateStrategies:
        if lr.__name__ == learning_rate_strategy:
            learning_rate_function = lr
            break
    if learning_rate_function is None:
        raise ValueError(f'Unknown learning rate strategy: {learning_rate_strategy}')

    x, y = sympy.symbols('x y')
    f: sympy.Expr = eval(
        function.replace('^', '**'),
        {
            'x': x,
            'y': y,
            'sp': sympy,
            **{name: getattr(sympy, name) for name in dir(sympy) if not name.startswith('_')}
        }
    )

    start_time = time.time()

    res = minimize_method(
        f,
        np.array([start_x, start_y]),
        learning_rate_function(learning_rate),
        max_steps,
        delta_xy_threshold if delta_xy else None,
        delta_f_threshold if delta_f else None
    )

    # время работы
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_time = round(elapsed_time % 1, 3)

    # рисуем график
    plotable_f = fs.from_sympy_to_plotable_func(f)
    x = np.linspace(x_lower_limit, x_upper_limit, 1000)
    y = np.linspace(y_lower_limit, y_upper_limit, 1000)
    X, Y = np.meshgrid(x, y)
    Z = plotable_f([X, Y])

    fig = go.Figure()
    # 3d поверхность
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, cmin=z_lower_limit, cmax=z_upper_limit))

    match res:
        case DescentOptimizationResult(_, _, _, _, path):
            path = np.array(path)
            # путь градиента
            for i, point in enumerate(path):
                fig.add_trace(go.Scatter3d(x=[point[0]], y=[point[1]], z=[plotable_f([point[0], point[1]])],
                                           mode='markers', showlegend=False,
                                           marker=dict(size=6), line=dict(width=5, color=get_color(i, len(path)))))
        case SimplexOptimizationResult(_, _, _, _, simplexes):
            for i, simplex in enumerate(simplexes):
                fig.add_trace(
                    go.Scatter3d(x=simplex[:, 0], y=simplex[:, 1], z=plotable_f([simplex[:, 0], simplex[:, 1]]),
                                 mode='lines+markers', showlegend=False, marker=dict(size=6),
                                 line=dict(width=5, color=get_color(i, len(simplexes)))))

    fig.update_layout(scene=dict(zaxis=dict(nticks=10, range=[z_lower_limit, z_upper_limit]),
                                 xaxis=dict(nticks=10, range=[x_lower_limit, x_upper_limit]),
                                 yaxis=dict(nticks=10, range=[y_lower_limit, y_upper_limit])),
                      margin=dict(l=0, r=0, b=0, t=0))

    return (fig,
            [
                f'функция: {function}',
                html.Br(),
                f'метод поиска минимума: {method_name}',
                html.Br(),
                f'метод выбора шага: {learning_rate_strategy}',
                html.Br(),
                f'критерий остановки: {res.stop_reason}',
                html.Br(),
                f'количество итераций: {res.iterations}',
                html.Br(),
                f'время выполнения: {elapsed_time} секунд',
                html.Br(),
                f'минимум функции: {res.result.tolist()}',

            ])


if __name__ == '__main__':
    app.run_server(debug=True)
