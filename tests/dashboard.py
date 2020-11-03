import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from masskrug.dashboard import nav, controls

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
external_stylesheets = ['https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
                        dbc.themes.BOOTSTRAP]

col = dbc.Col(children=[html.H5("Parameters"), nav, html.H5("Controls")], md=4)


class Dashboard:
    def __init__(self, *args, app=None, **kwargs):
        self.__app__ = app
        if app is None:
            self.__app__ = dash.Dash(*args, **kwargs)

        self.construct()
        self.register_callbacks()

    @property
    def app(self):
        return self.__app__

    def register_callbacks(self):
        @self.__app__.callback(
            Output("my-div", "children"),
            [Input(component_id='my-id', component_property='value')]
        )
        def call_back(input_value):
            return f'You\'ve entered {input_value}'

    def construct(self):
        self.app.layout = dbc.Container([
            dbc.Row([
                col,
                dbc.Col([
                    dcc.Input(id='my-id', value='initial value', type='text'),
                    html.Div(id='my-div')],
                    md=8)
            ])
        ], id="parameter_bar", style={"display": "grid"})


my_app = Dashboard(__name__, external_stylesheets=external_stylesheets)

if __name__ == '__main__':
    col.children.append(controls)
    print("test")
    my_app.app.run_server(debug=True, threaded=True)
