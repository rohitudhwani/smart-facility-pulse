"""
Sobha-inspired dark command center theme.
Premium-residential aesthetic: deep slate base, warm bronze + emerald accents.
Avoids the overused navy/gold dashboard cliche.
"""

# ===== BACKGROUND TONES =====
BG_PRIMARY = "#0F1419"       # main background — deep charcoal
BG_SECONDARY = "#1A1F2E"     # cards, panels — slate
BG_ELEVATED = "#252B3D"      # hover, raised elements
BORDER = "#2D3548"           # subtle dividers

# ===== TEXT =====
TEXT_PRIMARY = "#E8EAED"     # main text
TEXT_SECONDARY = "#9AA0A6"   # labels, metadata
TEXT_MUTED = "#5F6368"       # subdued

# ===== BRAND ACCENTS (Sobha-aligned, NOT navy/gold) =====
ACCENT_PRIMARY = "#C8956D"   # warm muted bronze (primary brand)
ACCENT_HOVER = "#D9A682"     # lighter bronze for hover

# ===== SEMANTIC (status) =====
STATUS_HEALTHY = "#10B981"   # emerald — good/normal
STATUS_WATCH = "#F59E0B"     # amber — watch/warning
STATUS_ALERT = "#EF4444"     # coral red — alert/critical
STATUS_OFFLINE = "#6B7280"   # gray — sensor down

# ===== DATA VIZ =====
VIZ_TEAL = "#06B6D4"         # cool data (humidity, water)
VIZ_BRONZE = "#C8956D"       # primary data
VIZ_PURPLE = "#A78BFA"       # secondary data series
VIZ_PINK = "#F472B6"         # tertiary data

# ===== STATUS HEATMAP (for floor heatmap) =====
HEATMAP_NORMAL = "#10B981"
HEATMAP_WATCH = "#F59E0B"
HEATMAP_ALERT = "#EF4444"
HEATMAP_OFFLINE = "#6B7280"


def get_global_css():
    """Return the global CSS string injected on every page."""
    return f"""
    <style>
        /* === Page base === */
        .stApp {{
            background-color: {BG_PRIMARY};
            color: {TEXT_PRIMARY};
        }}

        /* === Sidebar === */
        section[data-testid="stSidebar"] {{
            background-color: {BG_SECONDARY};
            border-right: 1px solid {BORDER};
        }}
        section[data-testid="stSidebar"] * {{
            color: {TEXT_PRIMARY};
        }}

        /* === Headings === */
        h1, h2, h3, h4 {{
            color: {TEXT_PRIMARY};
            font-weight: 600;
            letter-spacing: -0.01em;
        }}

        /* === KPI metric cards === */
        [data-testid="stMetric"] {{
            background-color: {BG_SECONDARY};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 18px 20px;
            transition: border-color 0.2s ease;
        }}
        [data-testid="stMetric"]:hover {{
            border-color: {ACCENT_PRIMARY};
        }}
        [data-testid="stMetricLabel"] {{
            color: {TEXT_SECONDARY} !important;
            font-size: 0.78rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT_PRIMARY} !important;
            font-size: 1.9rem !important;
            font-weight: 600 !important;
        }}

        /* === Buttons === */
        .stButton>button {{
            background-color: {ACCENT_PRIMARY};
            color: {BG_PRIMARY};
            border: none;
            border-radius: 8px;
            font-weight: 500;
            padding: 0.5rem 1.2rem;
            transition: background-color 0.15s ease;
        }}
        .stButton>button:hover {{
            background-color: {ACCENT_HOVER};
        }}

        /* === Selectboxes === */
        .stSelectbox>div>div {{
            background-color: {BG_SECONDARY};
            border: 1px solid {BORDER};
            color: {TEXT_PRIMARY};
        }}

        /* === Dataframe === */
        [data-testid="stDataFrame"] {{
            background-color: {BG_SECONDARY};
            border-radius: 8px;
        }}

        /* === Custom status pills === */
        .status-pill {{
            display: inline-block;
            padding: 3px 12px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.03em;
        }}
        .status-normal {{ background: {STATUS_HEALTHY}22; color: {STATUS_HEALTHY}; border: 1px solid {STATUS_HEALTHY}55; }}
        .status-watch {{ background: {STATUS_WATCH}22; color: {STATUS_WATCH}; border: 1px solid {STATUS_WATCH}55; }}
        .status-alert {{ background: {STATUS_ALERT}22; color: {STATUS_ALERT}; border: 1px solid {STATUS_ALERT}55; }}

        /* === Brand banner === */
        .brand-banner {{
            background: linear-gradient(135deg, {BG_SECONDARY} 0%, {BG_ELEVATED} 100%);
            border-left: 3px solid {ACCENT_PRIMARY};
            padding: 14px 22px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .brand-banner h1 {{
            margin: 0;
            font-size: 1.5rem;
            color: {TEXT_PRIMARY};
        }}
        .brand-banner p {{
            margin: 4px 0 0 0;
            color: {TEXT_SECONDARY};
            font-size: 0.85rem;
        }}

        /* === Hide Streamlit chrome === */
        #MainMenu, footer {{visibility: hidden;}}
    </style>
    """


# Plotly chart template — applies these colors to every chart automatically
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': BG_SECONDARY,
        'plot_bgcolor': BG_SECONDARY,
        'font': {'color': TEXT_PRIMARY, 'family': 'system-ui, -apple-system, sans-serif'},
        'xaxis': {'gridcolor': BORDER, 'zerolinecolor': BORDER, 'color': TEXT_SECONDARY},
        'yaxis': {'gridcolor': BORDER, 'zerolinecolor': BORDER, 'color': TEXT_SECONDARY},
        'colorway': [ACCENT_PRIMARY, VIZ_TEAL, STATUS_HEALTHY, VIZ_PURPLE, STATUS_WATCH, VIZ_PINK],
        'hoverlabel': {'bgcolor': BG_ELEVATED, 'bordercolor': ACCENT_PRIMARY,
                       'font': {'color': TEXT_PRIMARY}},
    }
}
