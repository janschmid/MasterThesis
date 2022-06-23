"""Visualize pandas dataframe with interactive images (mouseover)"""
import base64
import io
import os.path as p

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.io as pio
from dash import Dash, Input, Output, dcc, html, no_update
from PIL import Image
from sklearn.metrics import confusion_matrix
import os
import pyperclip
pio.renderers.default = "firefox"
# Set up the app now
app = Dash(__name__)

app.config["suppress_callback_exceptions"] = True
for i in range(10):
    @app.callback(
        Output("graph-tooltip-{0}".format(i), "show"),
        Output("graph-tooltip-{0}".format(i), "bbox"),
        Output("graph-tooltip-{0}".format(i), "children"),
        Output("graph-tooltip-{0}".format(i), "direction"),
        Input("graph-dcc-{0}".format(i), "hoverData"),
        # Input('graph-click-{0}'.format(i), component_property='n_clicks')
    )
    def display_hover(hover_data):
        """Called by app, don't call from outside"""
        return __display_hover(hover_data)
for i in range(10):
    @app.callback(
    Output('click-data-{0}'.format(i), 'children'),
    Input("graph-dcc-{0}".format(i), 'clickData')
    )
    def update_output(click_data):
        try:
            name = os.path.basename(click_data["points"][0]["customdata"][0])
            pyperclip.copy(name.replace("cam_", ""))
        except:
            pass

def __display_hover(hover_data):
    """Called by app, don't call from outside"""
    if hover_data is None:
        return False, no_update, no_update, no_update
    # Load image with pillow
    image_path = hover_data["points"][0]["customdata"][0]
    # print(image_path)
    im = Image.open(image_path)

    # dump it to base64
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image

    # demo only shows the first point, but other points may also be available
    hover_data = hover_data["points"][0]
    bbox = hover_data["bbox"]

    # control the position of the tooltip
    hover_data["y"]
    direction = "bottom"
    name = p.basename(image_path)
    children = [
        html.Img(
            src=im_url,
            style={"height": "250px"},
        ),
        html.P(name),
    ]
    return True, bbox, children, direction

def create_text_area(content, id=0):
    html_div = html.Div(
        dcc.Textarea(
        id='textarea-example-{0}'.format(id),
        value=content,
        style={'width': '100%', 'height': 300},
        readOnly=True
        ),
        # html.Div(id='textarea-example-output', style={'whiteSpace': 'pre-line'})
    )
    return html_div


def create_confustion_matrix(dataset, title, x_row_name, y_row_name, labels):
    """Thanks to https://stackoverflow.com/questions/60860121/plotly-how-to-make-an-annotated-confusion-matrix-using-a-heatmap
    Crate confusion matrix for plotly visualizer
    :param dataset: pandas dataset
    :param title: title of confusion matrix
    :param x_row_name: row in dataset for x_row
    :param y_row_name: row in dataset for y_row
    :return: html_div confusion matrix, can be passed to plot_html_figures
    """
    z = confusion_matrix(np.array(dataset[x_row_name]), np.array(dataset[y_row_name]))
    z=np.flip(z, axis=1)
    x=labels[::-1]
    y=labels
    z_text = [[str(y) for y in x] for x in z]
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    # add custom xaxis title
    # add title
    fig.update_layout(
        title_text="<i><b>{0}</b></i>".format(title),
        # xaxis = dict(title='x'),
        # yaxis = dict(title='x')
    )
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=0.5,
            y=-0.15,
            showarrow=False,
            text="Predicted value",
            xref="paper",
            yref="paper",
        )
    )

    # add custom yaxis title
    fig.add_annotation(
        dict(
            font=dict(color="black", size=14),
            x=-0.35,
            y=0.5,
            showarrow=False,
            text="Real value",
            textangle=-90,
            xref="paper",
            yref="paper",
        )
    )
    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig["data"][0]["showscale"] = True

    html_div_conf_matrix = html.Div(
        className="container",
        children=[
            # html.H1(children=title),
            dcc.Graph(id="graph-dcc-" + title, figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-" + title, direction="bottom"),
        ],
    )

    return html_div_conf_matrix


def create_line_plot(dataset, x_row_name, y_row_name, color_row_name, title):
    """Create line plot for plotly_visualizer
    :param title: title of plot
    :param x_row_name: row in dataset for x_row
    :param y_row_name: row in dataset for y_row
    :param color_row_name: row in dataset which defines color of points
    :param title: title of diagram
    :return: html_div confusion matrix, can be passed to plot_html_figures
    """

    fig = px.line(dataset, x=x_row_name, y=y_row_name, color=color_row_name)
    fig.update_layout(
        title_text="<i><b>{0}</b></i>".format(title),
        # xaxis = dict(title='x'),
        # yaxis = dict(title='x')
    )
    html_fig = html.Div(
        className="container",
        children=[
            # html.H1(children=title),
            dcc.Graph(id="graph-{0}-dcc".format(title), figure=fig, clear_on_unhover=True),
            dcc.Tooltip(id="graph-tooltip-{0}".format(title), direction="top"),
        ],
    )

    return html_fig


def create_scatter_plot(dataset, x_row_name, y_row_name, color_row_name, hover_row_name, title, id=0):
    """Create scatter plot for plotly_visualizer
    :param title: title of plot
    :param x_row_name: row in dataset for x_row
    :param y_row_name: row in dataset for y_row
    :param color_row_name: row in dataset which defines color of points
    :param hover_row_name: this row has to contain the path to a image file which is visualized when hovering, set to None if no hover data available
    :param title: title of diagram
    :param id: Each graph needs a unique id, id has to be between [0,9], for beyond 9, increase range of app callback
    :return: html_div confusion matrix, can be passed to plot_html_figures
    """
    if hover_row_name is None:
        fig = px.scatter(dataset, x=x_row_name, y=y_row_name, color=color_row_name)
        fig.update_layout(
            title_text="<i><b>{0}</b></i>".format(title),
            # xaxis = dict(title='x'),
            # yaxis = dict(title='x')
        )
        html_fig = html.Div(
            className="container",
            children=[
                # html.H1(children=title),
                dcc.Graph(id="graph-{0}-dcc".format(title), figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id="graph-tooltip-{0}".format(title), direction="top"),
            ],
        )
    else:
        fig = px.scatter(dataset, x=x_row_name, y=y_row_name, color=color_row_name, hover_data=[hover_row_name])
        fig.update_layout(clickmode='event+select')
        fig.update_traces(
            hoverinfo="none",
            hovertemplate=None,
            # marker=dict(size=30)
        )
        graph_id = "graph-dcc-" + str(id)
        click_id = "click-data-"+str(id)
        tooltip_id = "graph-tooltip-" + str(id)
        html_fig = html.Div(
            className="container",
            children=[
                html.H1(children=title),
                dcc.Graph(id=graph_id, figure=fig, clear_on_unhover=True),
                dcc.Tooltip(id=tooltip_id, direction="top"),
               html.Div([
                    dcc.Markdown(''),
                    html.Pre(id=click_id),
                ], className='three columns'), 
            ],
        )

    # app.callback(Output(tooltip_id, "direction"),
    # Input(graph_id, "hoverData"))
    return html_fig


def plot_html_figures(figures):
    """Setup the plot, call 'plotly_visualizer.app.runserver(debug=False)' to start visuliazation. Don't set debug to true, this works only when calling it inside from this file
    :param dataset: pandas dataframe to plot
    :param x_row_name: x row to plot from the pandas dataframe
    :param y_row_name: y row to plot from the pandas dataframe
    :param color_row_name: row of pandas dataframe which is used for color
    :param hover_row_name: row of pandas dataframe which is used for hover event, this has to contain the full link to the file path of the image"""
    app.layout = html.Div(figures, id="container")


if __name__ == "__main__":
    # test()
    broken = pd.read_csv(
        "data/color_thresholder_dssd/debug_broken.csv"
    )
    broken = broken.sort_values(by=["mean_normal"], ascending=False)
    # broken.to_csv(broken_csv_path, index=False, Header=False)
    unbroken = pd.read_csv(
        "data/color_thresholder_dssd/debug_unbroken.csv"
    )
    unbroken = unbroken.sort_values(by=["mean_normal"], ascending=False)
    broken["broken"] = 1
    unbroken["broken"] = 0
    combined = broken.append(unbroken)
    combined.reset_index(drop=True, inplace=True)
    figure = create_scatter_plot(combined, "mean_normal", "mean_shifted", "broken", "name", "debugSite")
    figure1 = create_scatter_plot(combined, "mean_normal", "mean_shifted", "broken", "name", "debugSite1", 1)
    plot_html_figures([figure, figure1])
    app.run_server(host="127.0.0.1", port="8080", debug=True)
