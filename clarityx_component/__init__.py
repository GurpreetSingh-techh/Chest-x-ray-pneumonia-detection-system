import streamlit.components.v1 as components
from pathlib import Path

_COMPONENT_DIR = Path(__file__).parent / "frontend"
clarityx_component = components.declare_component(
    "clarityx_component",
    path=str(_COMPONENT_DIR),
)

