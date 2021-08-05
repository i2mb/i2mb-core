import dash_bootstrap_components as dbc
import dash_html_components as html

"""
1. Percentage of the population with contact tracing list, i.e., as the population is generated, not all agents 
 are given the contact tracing property, and thus only contact tracing is possible between agents with the property. 
2. Life time of contacts on each particle's list 
3. Time to isolation 
4. Probability of successful contact exchange => to control false negative rate. 
5. Time in contact before exchange relative to time in contact for infection.
6. Additional contacts tracked without risk of infection (walls, doors, cars, etc.) => needs assumptions about
 probability of these situations (maybe from literature).
7. Considering the FP rate, prob. of success of isolation.
8. For completeness, probably without major effect: Assume some negative infection test probability for a test during
 isolation, admit individual back to population 
"""
tab1_content = [dbc.Card([
    dbc.CardHeader(html.H6("Contact tracing")),
    dbc.CardBody(
        [
            html.P("This is tab 1!", className="card-text"),
            dbc.Button("Click here", color="success"),
        ]
    )], className="mt-3"),
    dbc.Card([
        dbc.CardHeader(html.H6("Isolation")),
        dbc.CardBody(
            [
                html.P("This is tab 1!", className="card-text"),
                dbc.Button("Click here", color="success"),
            ]
        )], className="mt-3")]

tab2_content = dbc.Card(
    dbc.CardBody(
        [
            html.P("This is tab 2!", className="card-text"),
            dbc.Button("Don't click here", color="danger"),
        ]
    ),
    className="mt-3",
)

nav = dbc.Tabs(
    [
        dbc.Tab(tab1_content, label="Model"),
        dbc.Tab(tab2_content, label="Run")
    ]
)

controls = dbc.Card([
    dbc.FormGroup([
        dbc.Label(["Live Animation"]),
        html.Span([
            html.I(id='id-icon', className='fa fa-info'),
            dbc.Popover([
                dbc.PopoverHeader("Popover header"),
                dbc.PopoverBody("And here's some amazing content. Cool!"),
            ],
                id="popover",
                is_open=False,
                target="id-icon",
            )],
            style={"float": "right"}),
        dbc.Checklist(
            options=[
                {"label": "Enabled", "value": 1},
            ],
            value=[],
            id="live-animation",
            switch=True,
        ),
    ]
    )
],
    className="mt-3",
)
