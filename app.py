import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

# ========== CONFIGURATION DES POIDS PAR POSTE ==========

POSITION_SCORING = {
    "GK": {
        "description": "Gardiens de but",
        "categories": {
            "Saves & Shot Stopping": {
                "weight": 35,
                "columns": ["SoT", "Dist", "FK"],
                "inverse": []
            },
            "Distribution": {
                "weight": 25,
                "columns": ["Cmp%", "TotDist", "PrgDist"],
                "inverse": []
            },
            "Positioning & Command": {
                "weight": 20,
                "columns": ["Int", "Clr", "Err"],
                "inverse": ["Err"]
            },
            "Aerial Ability": {
                "weight": 15,
                "columns": ["Won", "Won%"],
                "inverse": []
            },
            "Concentration": {
                "weight": 5,
                "columns": ["CrdY", "CrdR"],
                "inverse": ["CrdY", "CrdR"]
            }
        }
    },
    "DF": {
        "description": "Défenseurs centraux",
        "categories": {
            "Defending": {
                "weight": 35,
                "columns": ["Tkl", "TklW", "Int", "Blocks", "Clr"],
                "inverse": []
            },
            "Aerial Duels": {
                "weight": 20,
                "columns": ["Won", "Won%"],
                "inverse": []
            },
            "Ball Playing": {
                "weight": 20,
                "columns": ["Cmp%", "PrgP", "PrgDist"],
                "inverse": []
            },
            "Positioning": {
                "weight": 15,
                "columns": ["Err", "Tkl%"],
                "inverse": ["Err"]
            },
            "Discipline": {
                "weight": 10,
                "columns": ["CrdY", "CrdR", "Fls"],
                "inverse": ["CrdY", "CrdR", "Fls"]
            }
        }
    },
    "DF,MF": {
        "description": "Défenseurs polyvalents / Latéraux",
        "categories": {
            "Defending": {
                "weight": 30,
                "columns": ["Tkl", "TklW", "Int", "Blocks"],
                "inverse": []
            },
            "Offensive Support": {
                "weight": 25,
                "columns": ["Ast", "xAG", "KP", "Crs", "PPA"],
                "inverse": []
            },
            "Ball Progression": {
                "weight": 25,
                "columns": ["PrgC", "PrgP", "Carries", "PrgDist"],
                "inverse": []
            },
            "Defensive Positioning": {
                "weight": 15,
                "columns": ["Won", "Recov"],
                "inverse": []
            },
            "Distribution": {
                "weight": 5,
                "columns": ["Cmp%", "TotDist"],
                "inverse": []
            }
        }
    },
    "MF,DF": {
        "description": "Milieux défensifs",
        "categories": {
            "Ball Recovery": {
                "weight": 35,
                "columns": ["Tkl", "TklW", "Int", "Recov"],
                "inverse": []
            },
            "Distribution": {
                "weight": 30,
                "columns": ["Cmp%", "PrgP", "PrgDist", "TotDist"],
                "inverse": []
            },
            "Defensive Positioning": {
                "weight": 20,
                "columns": ["Blocks", "Won%"],
                "inverse": []
            },
            "Discipline": {
                "weight": 10,
                "columns": ["CrdY", "CrdR", "Fls"],
                "inverse": ["CrdY", "CrdR", "Fls"]
            },
            "Ball Progression": {
                "weight": 5,
                "columns": ["PrgC", "Carries"],
                "inverse": []
            }
        }
    },
    "MF": {
        "description": "Milieux de terrain centraux",
        "categories": {
            "Passing & Distribution": {
                "weight": 30,
                "columns": ["Cmp%", "TotDist", "PrgDist", "PrgP"],
                "inverse": []
            },
            "Chance Creation": {
                "weight": 25,
                "columns": ["KP", "xAG", "Ast", "SCA"],
                "inverse": []
            },
            "Ball Progression": {
                "weight": 20,
                "columns": ["PrgC", "Carries", "Succ", "PrgR"],
                "inverse": []
            },
            "Defensive Contribution": {
                "weight": 15,
                "columns": ["Tkl", "Int", "Recov"],
                "inverse": []
            },
            "Shooting": {
                "weight": 10,
                "columns": ["Gls", "xG", "Sh", "SoT%"],
                "inverse": []
            }
        }
    },
    "MF,FW": {
        "description": "Milieux offensifs / Meneurs de jeu",
        "categories": {
            "Chance Creation": {
                "weight": 30,
                "columns": ["KP", "xAG", "Ast", "SCA", "GCA"],
                "inverse": []
            },
            "Dribbling & Take-Ons": {
                "weight": 25,
                "columns": ["Succ", "Succ%", "PrgC", "Mis"],
                "inverse": ["Mis"]
            },
            "Shooting": {
                "weight": 25,
                "columns": ["Gls", "xG", "Sh", "SoT%", "G/Sh"],
                "inverse": []
            },
            "Ball Progression": {
                "weight": 15,
                "columns": ["PrgP", "PrgR", "Carries"],
                "inverse": []
            },
            "Link-up Play": {
                "weight": 5,
                "columns": ["Cmp%", "TO"],
                "inverse": ["TO"]
            }
        }
    },
    "FW,MF": {
        "description": "Attaquants polyvalents / Seconds attaquants",
        "categories": {
            "Finishing": {
                "weight": 30,
                "columns": ["Gls", "xG", "SoT%", "G/Sh", "G/SoT"],
                "inverse": []
            },
            "Chance Creation": {
                "weight": 25,
                "columns": ["Ast", "xAG", "KP", "SCA"],
                "inverse": []
            },
            "Dribbling": {
                "weight": 20,
                "columns": ["Succ", "Succ%", "PrgC"],
                "inverse": []
            },
            "Movement & Positioning": {
                "weight": 15,
                "columns": ["Touches", "PrgR", "Carries"],
                "inverse": []
            },
            "Shooting Volume": {
                "weight": 10,
                "columns": ["Sh", "SoT", "Dist"],
                "inverse": []
            }
        }
    },
    "FW": {
        "description": "Attaquants purs / Buteurs",
        "categories": {
            "Finishing": {
                "weight": 40,
                "columns": ["Gls", "xG", "G/Sh", "G/SoT", "SoT%"],
                "inverse": []
            },
            "Shooting": {
                "weight": 25,
                "columns": ["Sh", "SoT", "Dist"],
                "inverse": []
            },
            "Positioning": {
                "weight": 15,
                "columns": ["Touches", "PrgR"],
                "inverse": []
            },
            "Aerial Ability": {
                "weight": 10,
                "columns": ["Won", "Won%"],
                "inverse": []
            },
            "Link-up": {
                "weight": 10,
                "columns": ["Ast", "KP", "Cmp%"],
                "inverse": []
            }
        }
    }
}

# ========== CONFIGURATION ==========
st.set_page_config(
    page_title="CIES Scouting Report - U21 Players",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== STYLES CSS CIES ==========
st.markdown("""
<style>
    /* Thème sombre général */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Titre principal style CIES */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #00FF85;
        text-align: center;
        padding: 2rem 0;
        text-transform: uppercase;
        letter-spacing: 3px;
        border-bottom: 3px solid #00FF85;
        margin-bottom: 2rem;
    }
    
    /* Sous-titre */
    .subtitle {
        font-size: 1.5rem;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Cartes de joueurs */
    .player-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        border-left: 4px solid #00FF85;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Score badge */
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00FF85 0%, #00CC6A 100%);
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        box-shadow: 0 2px 4px rgba(0, 255, 133, 0.3);
    }
    
    /* Stats row */
    .stats-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #3D3D3D;
    }
    
    /* Metric cards */
    .metric-card {
        background: #1E1E1E;
        border: 1px solid #3D3D3D;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00FF85;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #AAAAAA;
        text-transform: uppercase;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #00FF85 0%, #00CC6A 100%);
        color: #000000;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 700;
        border-radius: 25px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 255, 133, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ========== FONCTIONS UTILITAIRES ==========

@st.cache_data
def load_data():
    """Charge et prépare les données"""
    df = pd.read_csv('Joueurs_2004_2.csv', sep=';', encoding='utf-8-sig')
    
    # Nettoyage
    df.columns = df.columns.str.strip()
    
    # Conversion numériques
    numeric_cols = df.columns[df.columns.get_loc('Age'):]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.fillna(0)
    
    # Calcul dimensions radar
    df = calculate_radar_dimensions(df)
    
    # Score global
    df = calculate_global_score(df)
    
    return df

def calculate_radar_dimensions(df):
    """
    Calcule les 6 dimensions du radar CIES
    """
    
    # 1. Ground Defence
    df['Ground_Defence'] = (
        normalize_column(df, 'Tkl') * 0.4 +
        normalize_column(df, 'TklW') * 0.3 +
        normalize_column(df, 'Int') * 0.3
    )
    
    # 2. Aerial Play
    df['Aerial_Play'] = (
        normalize_column(df, 'Won') * 0.6 +
        normalize_column(df, 'Won%') * 0.4
    )
    
    # 3. Distribution
    df['Distribution'] = (
        normalize_column(df, 'Cmp%') * 0.4 +
        normalize_column(df, 'PrgDist') * 0.3 +
        normalize_column(df, 'PrgP') * 0.3
    )
    
    # 4. Chance Creation
    df['Chance_Creation'] = (
        normalize_column(df, 'KP') * 0.35 +
        normalize_column(df, 'xAG') * 0.35 +
        normalize_column(df, 'Ast') * 0.3
    )
    
    # 5. Take On
    df['Take_On'] = (
        normalize_column(df, 'Succ') * 0.4 +
        normalize_column(df, 'Succ%') * 0.3 +
        normalize_column(df, 'PrgC') * 0.3
    )
    
    # 6. Finishing
    df['Finishing'] = (
        normalize_column(df, 'Gls') * 0.4 +
        normalize_column(df, 'xG') * 0.3 +
        normalize_column(df, 'SoT%') * 0.3
    )
    
    # Percentiles (0-100)
    radar_dims = ['Ground_Defence', 'Aerial_Play', 'Distribution', 
                  'Chance_Creation', 'Take_On', 'Finishing']
    
    for dim in radar_dims:
        df[f'{dim}_Percentile'] = (df[dim].rank(pct=True) * 100).round(0)
    
    return df

def normalize_column(df, col_name):
    """Normalise une colonne entre 0 et 1"""
    if col_name not in df.columns:
        return pd.Series(0, index=df.index)
    
    col = df[col_name]
    min_val = col.min()
    max_val = col.max()
    
    if max_val == min_val:
        return pd.Series(0, index=df.index)
    
    return (col - min_val) / (max_val - min_val)

def calculate_global_score(df):
    """Calcule le score global sur 100"""
    radar_dims = ['Ground_Defence', 'Aerial_Play', 'Distribution', 
                  'Chance_Creation', 'Take_On', 'Finishing']
    
    df['Global_Score'] = df[radar_dims].mean(axis=1) * 100
    df['Global_Score'] = df['Global_Score'].round(1)
    
    return df

def estimate_value(score, age, league):
    """Estime la valeur du joueur en millions d'euros"""
    base_value = score / 10
    age_factor = 1 + (21 - age) * 0.1
    
    top_leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
    league_factor = 1.5 if league in top_leagues else 1.0
    
    value = base_value * age_factor * league_factor
    
    min_value = round(value * 0.8, 1)
    max_value = round(value * 1.2, 1)
    
    return f"€ {min_value}-{max_value} M"

def create_radar_chart(player_data, player_name):
    """Crée le radar chart style CIES"""
    
    categories = ['Ground\nDefence', 'Aerial\nPlay', 'Distribution', 
                  'Chance\nCreation', 'Take\nOn', 'Finishing']
    
    values = [
        player_data['Ground_Defence_Percentile'],
        player_data['Aerial_Play_Percentile'],
        player_data['Distribution_Percentile'],
        player_data['Chance_Creation_Percentile'],
        player_data['Take_On_Percentile'],
        player_data['Finishing_Percentile']
    ]
    
    values += values[:1]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0, 255, 133, 0.3)',
        line=dict(color='#00FF85', width=3),
        name=player_name
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(30, 30, 30, 0.8)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='',
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)',
                linecolor='rgba(255, 255, 255, 0.2)'
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12, family='Arial'),
        margin=dict(l=80, r=80, t=40, b=40),
        height=500
    )
    
    return fig

def create_comparison_radar(player1_data, player2_data, name1, name2):
    """Radar de comparaison pour 2 joueurs"""
    
    categories = ['Ground\nDefence', 'Aerial\nPlay', 'Distribution', 
                  'Chance\nCreation', 'Take\nOn', 'Finishing']
    
    values1 = [
        player1_data['Ground_Defence_Percentile'],
        player1_data['Aerial_Play_Percentile'],
        player1_data['Distribution_Percentile'],
        player1_data['Chance_Creation_Percentile'],
        player1_data['Take_On_Percentile'],
        player1_data['Finishing_Percentile']
    ]
    
    values2 = [
        player2_data['Ground_Defence_Percentile'],
        player2_data['Aerial_Play_Percentile'],
        player2_data['Distribution_Percentile'],
        player2_data['Chance_Creation_Percentile'],
        player2_data['Take_On_Percentile'],
        player2_data['Finishing_Percentile']
    ]
    
    values1 += values1[:1]
    values2 += values2[:1]
    categories_closed = categories + [categories[0]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values1,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(0, 255, 133, 0.3)',
        line=dict(color='#00FF85', width=3),
        name=name1
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=values2,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(255, 68, 68, 0.3)',
        line=dict(color='#FF4444', width=3),
        name=name2
    ))
    
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(30, 30, 30, 0.8)',
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                ticks='',
                gridcolor='rgba(255, 255, 255, 0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255, 255, 255, 0.2)'
            )
        ),
        showlegend=True,
        legend=dict(bgcolor='rgba(30, 30, 30, 0.8)', font=dict(color='white')),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        margin=dict(l=80, r=80, t=40, b=40),
        height=500
    )
    
    return fig

def display_player_card(player_data):
    """Affiche la carte d'identité du joueur"""
    st.markdown(f"""
    <div class="player-card">
        <h2 style="color: #00FF85; margin-bottom: 1rem;">🎯 {player_data['Player']}</h2>
        <div class="stats-row">
            <span style="color: #AAAAAA;">Position:</span>
            <span style="color: #FFFFFF; font-weight: 600;">{player_data['Pos']}</span>
        </div>
        <div class="stats-row">
            <span style="color: #AAAAAA;">Club:</span>
            <span style="color: #FFFFFF; font-weight: 600;">{player_data['Squad']}</span>
        </div>
        <div class="stats-row">
            <span style="color: #AAAAAA;">League:</span>
            <span style="color: #FFFFFF; font-weight: 600;">{player_data['Comp']}</span>
        </div>
        <div class="stats-row">
            <span style="color: #AAAAAA;">Age:</span>
            <span style="color: #FFFFFF; font-weight: 600;">{player_data['Age']} years (Born: {player_data['Born']})</span>
        </div>
        <div class="stats-row">
            <span style="color: #AAAAAA;">Nation:</span>
            <span style="color: #FFFFFF; font-weight: 600;">{player_data['Nation']}</span>
        </div>
        <div class="stats-row">
            <span style="color: #AAAAAA;">Minutes:</span>
            <span style="color: #FFFFFF; font-weight: 600;">{int(player_data['Min'])} min ({int(player_data['MP'])} matches)</span>
        </div>
        <div style="margin-top: 1.5rem; text-align: center;">
            <span class="score-badge">SCORE: {player_data['Global_Score']}/100</span>
        </div>
        <div style="margin-top: 1rem; text-align: center;">
            <p style="color: #00FF85; font-size: 1.1rem; margin: 0;">
                <strong>Est. Value:</strong> {estimate_value(player_data['Global_Score'], player_data['Age'], player_data['Comp'])}
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ========== CHARGEMENT ==========
df = load_data()

# ========== NAVIGATION ==========
st.markdown('<h1 class="main-title">⚽ CIES Scouting Report</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Best U21 Players Worldwide - Born 2004 or later</p>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Home", "📊 Rankings", "👤 Player Profile", "⚖️ Comparison", "🎯 By Category", "📈 Analytics"],
    label_visibility="visible"
)

# ========== PAGE: HOME ==========
if menu == "🏠 Home":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{len(df)}</div>
            <div class="metric-label">Total Players</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['Comp'].nunique()}</div>
            <div class="metric-label">Leagues</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{df['Squad'].nunique()}</div>
            <div class="metric-label">Clubs</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ## 📖 About This Report
    
    This report presents the **best U21 players worldwide** (born 2004+) based on their performance.
    
    ### 🎯 Methodology
    
    Players analyzed across **6 key dimensions**:
    
    1. **Ground Defence** - Tackles, interceptions
    2. **Aerial Play** - Aerial duels won
    3. **Distribution** - Passing accuracy, progressive passes
    4. **Chance Creation** - Key passes, assists, xAG
    5. **Take On** - Successful dribbles, carries
    6. **Finishing** - Goals, xG, shooting accuracy
    
    ### 📊 Performance Index
    
    Each player receives a **Global Score /100** based on percentile rankings across all dimensions.
    
    ### 💰 Transfer Values
    
    Estimated values (in €M) account for:
    - Performance level
    - Age & potential
    - League competitiveness
    
    ---
    
    *Data: FBref.com / StatsBomb • Style: CIES Football Observatory*
    """)
    
    st.markdown("---")
    st.markdown("### 🏆 Top 10 Players Overall")
    
    top10 = df.nlargest(10, 'Global_Score')[['Player', 'Pos', 'Squad', 'Comp', 'Age', 'Global_Score']]
    top10['Rank'] = range(1, 11)
    top10 = top10[['Rank', 'Player', 'Pos', 'Squad', 'Comp', 'Age', 'Global_Score']]
    
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Global_Score": st.column_config.ProgressColumn(
                "Score", format="%.1f", min_value=0, max_value=100
            ),
        }
    )

# ========== PAGE: RANKINGS ==========
elif menu == "📊 Rankings":
    st.markdown("## 📊 Player Rankings")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position_filter = st.multiselect(
            "🎯 Position",
            options=sorted(df['Pos'].unique()),
            default=None
        )
    
    with col2:
        league_filter = st.multiselect(
            "🏆 League",
            options=sorted(df['Comp'].unique()),
            default=None
        )
    
    with col3:
        min_score = st.slider("📈 Min Score", 0, 100, 50)
    
    col4, col5 = st.columns(2)
    
    with col4:
        min_minutes = st.slider("⏱️ Min Minutes", 0, int(df['Min'].max()), 90)
    
    with col5:
        search_name = st.text_input("🔍 Search Name", "")
    
    # Filtrage
    filtered_df = df.copy()
    
    if position_filter:
        filtered_df = filtered_df[filtered_df['Pos'].isin(position_filter)]
    
    if league_filter:
        filtered_df = filtered_df[filtered_df['Comp'].isin(league_filter)]
    
    filtered_df = filtered_df[filtered_df['Global_Score'] >= min_score]
    filtered_df = filtered_df[filtered_df['Min'] >= min_minutes]
    
    if search_name:
        filtered_df = filtered_df[filtered_df['Player'].str.contains(search_name, case=False, na=False)]
    
    sort_col = st.selectbox("Sort by", ['Global_Score', 'Age', 'Min', 'Gls', 'Ast'], index=0)
    filtered_df = filtered_df.sort_values(by=sort_col, ascending=False)
    
    st.markdown(f"### 🎯 Found **{len(filtered_df)}** players")
    
    display_df = filtered_df[['Player', 'Pos', 'Squad', 'Comp', 'Age', 'Min', 'Gls', 'Ast', 'Global_Score']].copy()
    display_df['Rank'] = range(1, len(display_df) + 1)
    display_df = display_df[['Rank', 'Player', 'Pos', 'Squad', 'Comp', 'Age', 'Min', 'Gls', 'Ast', 'Global_Score']]
    
    display_df['Est. Value'] = display_df.apply(
        lambda row: estimate_value(row['Global_Score'], row['Age'], row['Comp']), axis=1
    )
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Global_Score": st.column_config.ProgressColumn("Score", format="%.1f", min_value=0, max_value=100),
        },
        height=600
    )
    
    csv = display_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download CSV", csv, 'scouting_report.csv', 'text/csv')

# ========== PAGE: PLAYER PROFILE ==========
elif menu == "👤 Player Profile":
    st.markdown("## 👤 Player Scouting Report")
    st.markdown("---")
    
    player_name = st.selectbox("Select a player", sorted(df['Player'].unique()), index=0)
    player_data = df[df['Player'] == player_name].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        display_player_card(player_data)
        
        st.markdown("---")
        st.markdown("### 📊 Key Stats")
        
        stats = {
            "Goals": int(player_data['Gls']),
            "Assists": int(player_data['Ast']),
            "xG": round(player_data['xG'], 2),
            "xAG": round(player_data['xAG'], 2),
            "Shots": int(player_data['Sh']),
            "SoT%": round(player_data['SoT%'], 1),
            "Pass Acc%": round(player_data['Cmp%'], 1),
            "Tackles": int(player_data['Tkl']),
            "Interceptions": int(player_data['Int']),
            "Dribbles": int(player_data['Succ']),
        }
        
        for k, v in stats.items():
            st.markdown(f"""<div class="stats-row">
                <span style="color: #AAA;">{k}:</span>
                <span style="color: #00FF85; font-weight: 600;">{v}</span>
            </div>""", unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Performance Radar")
        st.plotly_chart(create_radar_chart(player_data, player_name), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📈 Dimension Breakdown")
        
        dims = {
            "Ground Defence": player_data['Ground_Defence_Percentile'],
            "Aerial Play": player_data['Aerial_Play_Percentile'],
            "Distribution": player_data['Distribution_Percentile'],
            "Chance Creation": player_data['Chance_Creation_Percentile'],
            "Take On": player_data['Take_On_Percentile'],
            "Finishing": player_data['Finishing_Percentile']
        }
        
        for name, val in dims.items():
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.progress(int(val) / 100)
            with col_b:
                st.markdown(f"**{int(val)}**th")
            st.markdown(f"<small style='color: #AAA;'>{name}</small><br>", unsafe_allow_html=True)

# ========== PAGE: COMPARISON ==========
elif menu == "⚖️ Comparison":
    st.markdown("## ⚖️ Player Comparison")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        p1 = st.selectbox("Select Player 1", sorted(df['Player'].unique()), index=0, key='p1')
    
    with col2:
        p2 = st.selectbox("Select Player 2", sorted(df['Player'].unique()), index=1, key='p2')
    
    if p1 and p2:
        p1_data = df[df['Player'] == p1].iloc[0]
        p2_data = df[df['Player'] == p2].iloc[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            display_player_card(p1_data)
        
        with col2:
            display_player_card(p2_data)
        
        st.markdown("---")
        st.markdown("### 🎯 Performance Comparison")
        
        st.plotly_chart(create_comparison_radar(p1_data, p2_data, p1, p2), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Stats Comparison")
        
        comp = {
            "Metric": ["Score", "Age", "Min", "Gls", "Ast", "xG", "xAG", "Pass%", "Tkl", "Int", "Dribbles", "SoT%"],
            p1: [
                p1_data['Global_Score'], p1_data['Age'], int(p1_data['Min']),
                int(p1_data['Gls']), int(p1_data['Ast']), round(p1_data['xG'], 2),
                round(p1_data['xAG'], 2), round(p1_data['Cmp%'], 1),
                int(p1_data['Tkl']), int(p1_data['Int']), int(p1_data['Succ']),
                round(p1_data['SoT%'], 1)
            ],
            p2: [
                p2_data['Global_Score'], p2_data['Age'], int(p2_data['Min']),
                int(p2_data['Gls']), int(p2_data['Ast']), round(p2_data['xG'], 2),
                round(p2_data['xAG'], 2), round(p2_data['Cmp%'], 1),
                int(p2_data['Tkl']), int(p2_data['Int']), int(p2_data['Succ']),
                round(p2_data['SoT%'], 1)
            ]
        }
        
        st.dataframe(pd.DataFrame(comp), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🏆 Verdict")
        
        if p1_data['Global_Score'] > p2_data['Global_Score']:
            winner, diff = p1, p1_data['Global_Score'] - p2_data['Global_Score']
        else:
            winner, diff = p2, p2_data['Global_Score'] - p1_data['Global_Score']
        
        st.markdown(f"""<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #1E1E1E, #2D2D2D); border-radius: 10px; border: 2px solid #00FF85;">
            <h3 style="color: #00FF85; margin-bottom: 1rem;">⭐ {winner} leads by {diff:.1f} points!</h3>
            <p style="color: #AAA;">Based on overall performance</p>
        </div>""", unsafe_allow_html=True)

# ========== PAGE: BY CATEGORY ==========
elif menu == "🎯 By Category":
    st.markdown("## 🎯 Best Players by Category")
    st.markdown("---")
    
    st.markdown("""
    Find the best players in specific categories for each position.
    Each position has unique categories based on its requirements.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        position_selected = st.selectbox(
            "📍 Select Position",
            options=sorted([pos for pos in df['Pos'].unique() if pos in POSITION_SCORING]),
            key='cat_pos'
        )
    
    with col2:
        if position_selected and position_selected in POSITION_SCORING:
            categories = list(POSITION_SCORING[position_selected]['categories'].keys())
            category_selected = st.selectbox("🎯 Select Category", options=categories)
    
    if position_selected and category_selected:
        st.markdown(f"### 🏆 Top 20 {position_selected} in {category_selected}")
        
        score_col = f"{category_selected}_Score"
        
        if score_col in df.columns:
            position_df = df[df['Pos'] == position_selected].copy()
            
            # Filtrer les joueurs avec temps de jeu minimum
            min_minutes_cat = st.slider("Min Minutes Played", 0, 2000, 270, key='min_cat')
            position_df = position_df[position_df['Min'] >= min_minutes_cat]
            
            if len(position_df) > 0:
                top_category = position_df.nlargest(20, score_col)[
                    ['Player', 'Squad', 'Comp', 'Age', 'Min', score_col, 'Global_Score']
                ].copy()
                
                top_category['Rank'] = range(1, len(top_category) + 1)
                top_category = top_category[['Rank', 'Player', 'Squad', 'Comp', 'Age', 'Min', score_col, 'Global_Score']]
                
                st.dataframe(
                    top_category,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        score_col: st.column_config.ProgressColumn(
                            category_selected,
                            format="%.1f",
                            min_value=0,
                            max_value=100
                        ),
                        "Global_Score": st.column_config.ProgressColumn(
                            "Overall",
                            format="%.1f",
                            min_value=0,
                            max_value=100
                        ),
                    }
                )
                
                # Distribution des scores pour cette catégorie
                st.markdown("---")
                st.markdown(f"### 📊 Score Distribution - {category_selected}")
                
                fig_dist = px.histogram(
                    position_df,
                    x=score_col,
                    nbins=20,
                    title=f'{category_selected} Score Distribution for {position_selected}',
                    labels={score_col: 'Score', 'count': 'Number of Players'},
                    color_discrete_sequence=['#00FF85']
                )
                
                fig_dist.update_layout(
                    paper_bgcolor='#0E1117',
                    plot_bgcolor='#1E1E1E',
                    font=dict(color='#FFFFFF'),
                    xaxis=dict(gridcolor='#3D3D3D'),
                    yaxis=dict(gridcolor='#3D3D3D')
                )
                
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Stats
                st.markdown("### 📈 Category Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average", f"{position_df[score_col].mean():.1f}")
                with col2:
                    st.metric("Median", f"{position_df[score_col].median():.1f}")
                with col3:
                    st.metric("Max", f"{position_df[score_col].max():.1f}")
                with col4:
                    st.metric("Std Dev", f"{position_df[score_col].std():.1f}")
            else:
                st.warning("⚠️ No players found with the specified criteria")
        else:
            st.error(f"❌ Category score column '{score_col}' not found")

# ========== PAGE: ANALYTICS ==========
elif menu == "📈 Analytics":
    st.markdown("## 📈 Advanced Analytics")
    st.markdown("---")
    
    # Score distribution par position
    st.markdown("### 📊 Score Distribution by Position")
    
    positions_to_analyze = st.multiselect(
        "Select positions to analyze",
        options=sorted([pos for pos in df['Pos'].unique() if pos in POSITION_SCORING]),
        default=list(POSITION_SCORING.keys())[:4]
    )
    
    if positions_to_analyze:
        fig_box = go.Figure()
        
        for position in positions_to_analyze:
            position_df = df[df['Pos'] == position]
            
            fig_box.add_trace(go.Box(
                y=position_df['Global_Score'],
                name=position,
                boxmean='sd'
            ))
        
        fig_box.update_layout(
            title='Score Distribution by Position',
            yaxis_title='Score',
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E1E',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='#3D3D3D'),
            yaxis=dict(gridcolor='#3D3D3D'),
            showlegend=False
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    
    # Statistiques par position
    st.markdown("### 📈 Statistics by Position")
    
    stats_by_position = []
    for position in sorted([pos for pos in df['Pos'].unique() if pos in POSITION_SCORING]):
        position_df = df[df['Pos'] == position]
        if len(position_df) > 0:
            stats_by_position.append({
                'Position': position,
                'Description': POSITION_SCORING[position]['description'],
                'Count': len(position_df),
                'Avg Score': position_df['Global_Score'].mean(),
                'Median': position_df['Global_Score'].median(),
                'Max': position_df['Global_Score'].max(),
                'Std Dev': position_df['Global_Score'].std()
            })
    
    stats_df = pd.DataFrame(stats_by_position)
    stats_df = stats_df.round(1)
    
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Top leagues
    st.markdown("### 🏆 Top Leagues by Average Score")
    
    league_stats = df.groupby('Comp').agg({
        'Global_Score': 'mean',
        'Player': 'count'
    }).round(1)
    league_stats.columns = ['Avg Score', 'Players']
    league_stats = league_stats[league_stats['Players'] >= 10].sort_values('Avg Score', ascending=False).head(15)
    
    fig_leagues = px.bar(
        league_stats.reset_index(),
        x='Comp',
        y='Avg Score',
        title='Average Player Score by League (min. 10 players)',
        labels={'Comp': 'League', 'Avg Score': 'Average Score'},
        color='Avg Score',
        color_continuous_scale='Viridis'
    )
    
    fig_leagues.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF'),
        xaxis=dict(gridcolor='#3D3D3D', tickangle=-45),
        yaxis=dict(gridcolor='#3D3D3D')
    )
    
    st.plotly_chart(fig_leagues, use_container_width=True)
    
    st.markdown("---")
    
    # Age vs Score
    st.markdown("### 📊 Age vs Performance")
    
    fig_scatter = px.scatter(
        df[df['Min'] >= 270],
        x='Age',
        y='Global_Score',
        color='Pos',
        size='Min',
        hover_data=['Player', 'Squad', 'Comp'],
        title='Age vs Score (min. 270 minutes played)',
        labels={'Age': 'Age', 'Global_Score': 'Score', 'Pos': 'Position'},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig_scatter.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF'),
        xaxis=dict(gridcolor='#3D3D3D'),
        yaxis=dict(gridcolor='#3D3D3D')
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""<div style="text-align: center; padding: 2rem; color: #666;">
    <p style="margin: 0; font-size: 0.9rem;">
        ⚽ <strong>CIES Football Observatory</strong> - Scouting Report<br>
        <em>Data-driven player analysis</em>
    </p>
    <p style="margin-top: 1rem; font-size: 0.8rem; color: #444;">
        Data: FBref.com / StatsBomb • Players born 2004+<br>
        Methodology inspired by CIES Football Observatory
    </p>
</div>""", unsafe_allow_html=True)
