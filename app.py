import numpy as np
import pandas as pd
from vector_handler import model_handler
import json
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget 
from shinyswatch import theme
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            .navbar {
                flex-wrap: wrap !important;
                position: relative !important; /* Prevent overlap */
            }
            .content-area {
                padding-top: 60px !important; /* Adjust depending on navbar height */
            }
            """
        )
    ),
    ui.page_navbar(
        ui.nav_panel('Instructions', 
            ui.include_html('explanation_text.html')
        ),
        ui.nav_panel('Search',
            ui.input_text("search_text", "Search term(s)"),
            ui.input_action_button("search_button", "Search"),
        )
    )
)

model_handle = model_handler(
                    classifier_path,
                    doc_vec_path,
                    reference_df_path,
                    doc_vec_3D_path,
                    doc_vec_2D_path,
                    umap2D_path,
                    umap3D_path,
                    genre_ids_path,
                    country_ids_path,
                    time_ids_path,
                    genocide_ids_path,
                    tokenizer_path=None,
                    model_path=None,
                )

def server(input, output, session):
    # Initialize firebase vars
    key_str = "firebase_key.json"
    if not key_str:
        raise RuntimeError("FIREBASE_KEY not set in environment")
    with open(key_str) as f:
        key_dict = json.load(f)
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)
    db = firestore.client()

    
    
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()