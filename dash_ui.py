# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:46:12 2022

@author: rspet
"""
import dash
from dash import html
from dash import dcc
import plotly.graph_objects as go
import plotly.express as px
import stat_charts

dash_app = dash.Dash()

demo_data = stat_charts.get_demo_data("resources")
dfs  = stat_charts.create_stat_dfs(demo_data)
if __name__ == '__main__': 
    #dash_app.run_server()
    print("3")