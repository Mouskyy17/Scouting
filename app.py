import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

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
st.markdown('<h1 class="main-title">⚽ U21 Scouting Report</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Best U21 Players Worldwide - Born 2004 or later</p>', unsafe_allow_html=True)

menu = st.sidebar.radio(
    "📋 Navigation",
    ["🏠 Home", "📊 Rankings", "👤 Player Profile", "⚖️ Comparison"],
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
