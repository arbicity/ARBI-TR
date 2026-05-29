#!/bin/bash
# Start the ARBI-TR Streamlit frontend.
exec uv run streamlit run app.py --server.address=0.0.0.0 --server.port=8501
