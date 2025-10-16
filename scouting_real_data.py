import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Scouting Report U21 - Football Observatory",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        color: white;
    }
    .metric-container h3 {
        color: white;
        margin-top: 0;
    }
    .metric-container h1 {
        color: white !important;
        margin: 10px 0;
    }
    .metric-container p {
        color: rgba(255, 255, 255, 0.9);
    }
    .player-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 10px 0;
        border: 1px solid #e0e0e0;
    }
    .player-card h2, .player-card h3 {
        color: #1e3c72;
        margin-top: 0;
    }
    .player-card p {
        color: #2c3e50;
        margin: 8px 0;
        font-size: 0.95em;
    }
    .player-card strong {
        color: #1e3c72;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    div[style*="background: #f8f9fa"] {
        background: linear-gradient(135deg, #e0e7ff 0%, #cfd9df 100%) !important;
        padding: 15px;
        border-radius: 8px;
        margin: 5px 0;
        color: #1e3c72 !important;
    }
    div[style*="background: #f8f9fa"] strong {
        color: #1e3c72 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour mapper les positions aux catégories
def map_position_to_category(pos):
    pos = str(pos).upper()
    if 'GK' in pos:
        return 'Goalkeeper'
    elif pos in ['DF', 'DF,MF']:
        return 'Centre Back'
    elif 'DF' in pos:
        return 'Full/Wing Back'
    elif pos in ['MF', 'MF,DF']:
        return 'Midfielder'
    elif pos in ['MF,FW', 'FW,MF']:
        return 'Attacking Midfielder/Winger'
    elif 'FW' in pos:
        return 'Forward'
    else:
        return 'Midfielder'

# Fonction pour convertir en numérique de manière sécurisée
def safe_numeric(series):
    """Convertit une série en numérique en gérant les erreurs"""
    return pd.to_numeric(series, errors='coerce').fillna(0)

# Chargement des données réelles
@st.cache_data
def load_data():
    try:
        # Charger le CSV
        df = pd.read_csv('Joueurs_2004.csv', sep=';', encoding='utf-8')
        
        # Nettoyer les noms de colonnes (supprimer les espaces)
        df.columns = df.columns.str.strip()
        
        # Convertir toutes les colonnes numériques nécessaires
        numeric_columns = ['Age', 'Min', 'Gls', 'Ast', 'CrdY', 'CrdR', 'MP', 'Starts',
                          'xG', 'xAG', 'Tkl', 'Int', 'Won', 'Cmp%', 'KP', 'Succ', 'SoT']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = safe_numeric(df[col])
        
        # Créer le dataframe avec les colonnes nécessaires
        processed_df = pd.DataFrame()
        
        # Mapping des colonnes de base
        processed_df['name'] = df['Player'].astype(str)
        processed_df['age'] = df['Age']
        processed_df['club'] = df['Squad'].astype(str)
        processed_df['league'] = df['Comp'].astype(str)
        processed_df['country'] = df['Nation'].astype(str)
        processed_df['category'] = df['Pos'].astype(str).apply(map_position_to_category)
        
        # Statistiques de jeu
        processed_df['minutes_played'] = df['Min']
        processed_df['goals'] = df['Gls']
        processed_df['assists'] = df['Ast']
        processed_df['yellow_cards'] = df['CrdY']
        processed_df['red_cards'] = df['CrdR']
        processed_df['matches_played'] = df['MP']
        processed_df['starts'] = df['Starts']
        
        # Calcul de l'indice de performance (basé sur plusieurs métriques)
        # Normaliser les stats pour avoir un score sur 100
        processed_df['performance_index'] = (
            (df['Gls'] * 3 + 
             df['Ast'] * 2 + 
             df['Min'] / 30 +
             df['xG'] * 2 +
             df['xAG'] * 2) / 2
        ).clip(0, 100)
        
        # Pour les joueurs avec peu de stats, ajuster le score
        processed_df.loc[processed_df['minutes_played'] < 100, 'performance_index'] *= 0.5
        
        # Estimation de la valeur de transfert basée sur les performances
        base_value = 5 + (processed_df['performance_index'] / 10)
        age_factor = (21 - processed_df['age']) * 2
        processed_df['transfer_value_min'] = (base_value + age_factor).clip(1, 50)
        processed_df['transfer_value_max'] = (processed_df['transfer_value_min'] * 2).clip(5, 100)
        processed_df['transfer_value_avg'] = (processed_df['transfer_value_min'] + processed_df['transfer_value_max']) / 2
        
        # Attributs techniques (basés sur les stats du CSV)
        # Ground Defence (tacles, interceptions)
        processed_df['ground_defence'] = (
            (df['Tkl'] * 2 + df['Int']) * 5
        ).clip(0, 100)
        
        # Aerial Play (duels aériens gagnés)
        processed_df['aerial_play'] = (
            df['Won'] * 3
        ).clip(0, 100)
        
        # Distribution (passes complétées, précision)
        processed_df['distribution'] = df['Cmp%'].clip(0, 100)
        
        # Chance Creation (passes décisives, passes clés)
        processed_df['chance_creation'] = (
            (df['Ast'] * 5 + df['KP'] * 3)
        ).clip(0, 100)
        
        # Take On (dribbles réussis)
        processed_df['take_on'] = (
            df['Succ'] * 10
        ).clip(0, 100)
        
        # Finishing (buts, tirs cadrés)
        processed_df['finishing'] = (
            (df['Gls'] * 5 + df['SoT'] * 2)
        ).clip(0, 100)
        
        # Filtrer les joueurs avec au moins quelques données et s'assurer qu'il n'y a pas de valeurs NaN
        processed_df = processed_df[processed_df['minutes_played'] > 0].copy()
        processed_df = processed_df.fillna(0)
        
        return processed_df
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame()

# Classe pour le modèle de Machine Learning
class PlayerPotentialModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.is_fitted = False
    
    def train(self, df):
        features = ['age', 'performance_index', 'ground_defence', 'aerial_play', 
                   'distribution', 'chance_creation', 'take_on', 'finishing',
                   'minutes_played', 'goals', 'assists']
        
        X = df[features].fillna(0)
        y = df['transfer_value_avg']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.kmeans.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict_potential(self, player_data):
        if not self.is_fitted:
            return None
        
        if isinstance(player_data, np.ndarray):
            X = self.scaler.transform(player_data.reshape(1, -1))
        else:
            X = self.scaler.transform(np.array(player_data).reshape(1, -1))
        
        potential_score = self.model.predict(X)[0]
        
        age_factor = max(0, (21 - player_data[0]) / 4)
        performance_factor = player_data[1] / 100
        
        potential = (potential_score * 0.4 + age_factor * 30 + performance_factor * 30)
        return min(100, max(0, potential))
    
    def get_similar_players(self, player_data, df, n_similar=5):
        if not self.is_fitted:
            return pd.DataFrame()
        
        features = ['age', 'performance_index', 'ground_defence', 'aerial_play', 
                   'distribution', 'chance_creation', 'take_on', 'finishing',
                   'minutes_played', 'goals', 'assists']
        
        X = df[features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        if isinstance(player_data, np.ndarray):
            player_scaled = self.scaler.transform(player_data.reshape(1, -1))
        else:
            player_scaled = self.scaler.transform(np.array(player_data).reshape(1, -1))
        
        distances = np.sum((X_scaled - player_scaled) ** 2, axis=1)
        
        similar_indices = np.argsort(distances)[1:n_similar+1]
        return df.iloc[similar_indices]

# Fonction pour créer un radar chart
def create_radar_chart(player_data, player_name):
    categories = ['Ground Defence', 'Aerial Play', 'Distribution', 
                  'Chance Creation', 'Take On', 'Finishing']
    
    values = [
        player_data['ground_defence'],
        player_data['aerial_play'],
        player_data['distribution'],
        player_data['chance_creation'],
        player_data['take_on'],
        player_data['finishing']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name=player_name,
        line=dict(color='#2a5298', width=2),
        fillcolor='rgba(42, 82, 152, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title=f"Profil de performance - {player_name}",
        height=400
    )
    
    return fig

# Interface principale
def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <h1>⚽ SCOUTING REPORT - MEILLEURS JOUEURS U21</h1>
        <p>Analyse des performances et potentiel des jeunes talents mondiaux (Génération 2004+)</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    
    if df.empty:
        st.error("❌ Impossible de charger les données. Veuillez vérifier que le fichier 'Joueurs_2004.csv' est présent.")
        return
    
    st.success(f"✅ {len(df)} joueurs chargés avec succès!")
    
    # Initialisation du modèle ML
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = PlayerPotentialModel().train(df)
    
    # Sidebar pour les filtres
    st.sidebar.header("🔍 Filtres de recherche")
    
    # Filtres
    selected_category = st.sidebar.selectbox(
        "Catégorie de joueur",
        ['Toutes'] + sorted(df['category'].unique())
    )
    
    selected_league = st.sidebar.selectbox(
        "Championnat",
        ['Tous'] + sorted(df['league'].unique())
    )
    
    age_range = st.sidebar.slider(
        "Âge",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=(int(df['age'].min()), int(df['age'].max()))
    )
    
    min_performance = st.sidebar.slider(
        "Performance minimale",
        min_value=0,
        max_value=100,
        value=30
    )
    
    min_minutes = st.sidebar.slider(
        "Minutes minimum jouées",
        min_value=0,
        max_value=int(df['minutes_played'].max()),
        value=100,
        step=100
    )
    
    # Application des filtres
    filtered_df = df.copy()
    
    if selected_category != 'Toutes':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    
    if selected_league != 'Tous':
        filtered_df = filtered_df[filtered_df['league'] == selected_league]
    
    filtered_df = filtered_df[
        (filtered_df['age'] >= age_range[0]) & 
        (filtered_df['age'] <= age_range[1]) &
        (filtered_df['performance_index'] >= min_performance) &
        (filtered_df['minutes_played'] >= min_minutes)
    ]
    
    # Métriques globales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Joueurs analysés", len(filtered_df))
    
    with col2:
        avg_performance = filtered_df['performance_index'].mean()
        st.metric("Performance moyenne", f"{avg_performance:.1f}/100")
    
    with col3:
        avg_value = filtered_df['transfer_value_avg'].mean()
        st.metric("Valeur moyenne", f"€{avg_value:.1f}M")
    
    with col4:
        avg_age = filtered_df['age'].mean()
        st.metric("Âge moyen", f"{avg_age:.1f} ans")
    
    # Onglets principaux
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Top Joueurs", 
        "📊 Analyses", 
        "🎯 Joueur Détaillé", 
        "🤖 Prédictions IA"
    ])
    
    with tab1:
        st.header("🏆 Classement des meilleurs joueurs")
        
        # Tri par performance
        top_players = filtered_df.nlargest(20, 'performance_index')
        
        for idx, (_, player) in enumerate(top_players.iterrows(), 1):
            with st.container():
                st.markdown(f"""
                <div class="player-card">
                    <h3>#{idx} {player['name']}</h3>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p><strong>Club:</strong> {player['club']} ({player['league']})</p>
                            <p><strong>Pays:</strong> {player['country']}</p>
                            <p><strong>Âge:</strong> {player['age']:.0f} ans</p>
                            <p><strong>Catégorie:</strong> {player['category']}</p>
                        </div>
                        <div>
                            <p><strong>Performance:</strong> {player['performance_index']:.1f}/100</p>
                            <p><strong>Minutes:</strong> {player['minutes_played']:.0f}</p>
                            <p><strong>Buts:</strong> {player['goals']:.0f} | <strong>Passes:</strong> {player['assists']:.0f}</p>
                            <p><strong>Valeur estimée:</strong> €{player['transfer_value_min']:.1f}M - €{player['transfer_value_max']:.1f}M</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.header("📊 Analyses statistiques")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution par catégorie
            category_counts = filtered_df['category'].value_counts()
            fig_cat = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title="Distribution par catégorie",
                labels={'x': 'Nombre de joueurs', 'y': 'Catégorie'}
            )
            fig_cat.update_layout(height=500)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Relation performance vs valeur
            fig_scatter = px.scatter(
                filtered_df,
                x='performance_index',
                y='transfer_value_avg',
                color='category',
                size='minutes_played',
                hover_data=['name', 'club', 'goals', 'assists'],
                title="Performance vs Valeur de transfert"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top buteurs et passeurs
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🥅 Top 10 Buteurs")
            top_scorers = filtered_df.nlargest(10, 'goals')[['name', 'club', 'goals', 'matches_played']]
            for idx, (_, player) in enumerate(top_scorers.iterrows(), 1):
                st.write(f"{idx}. **{player['name']}** ({player['club']}) - {player['goals']:.0f} buts en {player['matches_played']:.0f} matchs")
        
        with col2:
            st.subheader("🎯 Top 10 Passeurs")
            top_assisters = filtered_df.nlargest(10, 'assists')[['name', 'club', 'assists', 'matches_played']]
            for idx, (_, player) in enumerate(top_assisters.iterrows(), 1):
                st.write(f"{idx}. **{player['name']}** ({player['club']}) - {player['assists']:.0f} passes en {player['matches_played']:.0f} matchs")
        
        # Analyse par championnat
        st.subheader("Performance par championnat")
        league_performance = filtered_df.groupby('league').agg({
            'performance_index': 'mean',
            'transfer_value_avg': 'mean',
            'name': 'count',
            'goals': 'sum',
            'assists': 'sum'
        }).round(2)
        league_performance.columns = ['Performance moyenne', 'Valeur moyenne (€M)', 'Nombre de joueurs', 'Total buts', 'Total passes']
        st.dataframe(league_performance.sort_values('Performance moyenne', ascending=False))
    
    with tab3:
        st.header("🎯 Analyse détaillée d'un joueur")
        
        # Sélection du joueur
        selected_player_name = st.selectbox(
            "Choisir un joueur",
            sorted(filtered_df['name'].unique())
        )
        
        player_data = filtered_df[filtered_df['name'] == selected_player_name].iloc[0]
        
        # Informations générales
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="player-card">
                <h2>{player_data['name']}</h2>
                <p><strong>Club:</strong> {player_data['club']}</p>
                <p><strong>Championnat:</strong> {player_data['league']}</p>
                <p><strong>Pays:</strong> {player_data['country']}</p>
                <p><strong>Âge:</strong> {player_data['age']:.0f} ans</p>
                <p><strong>Catégorie:</strong> {player_data['category']}</p>
                <p><strong>Performance:</strong> {player_data['performance_index']:.1f}/100</p>
                <p><strong>Valeur estimée:</strong> €{player_data['transfer_value_min']:.1f}M - €{player_data['transfer_value_max']:.1f}M</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Radar chart
            radar_fig = create_radar_chart(player_data, player_data['name'])
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Statistiques détaillées
        st.subheader("Statistiques détaillées")
        
        stats_cols = st.columns(4)
        
        with stats_cols[0]:
            st.metric("Matchs", f"{player_data['matches_played']:.0f}")
            st.metric("Titularisations", f"{player_data['starts']:.0f}")
        
        with stats_cols[1]:
            st.metric("Minutes jouées", f"{player_data['minutes_played']:.0f}")
            st.metric("Buts", f"{player_data['goals']:.0f}")
        
        with stats_cols[2]:
            st.metric("Passes décisives", f"{player_data['assists']:.0f}")
            if player_data['minutes_played'] > 0:
                goals_per_90 = (player_data['goals'] / player_data['minutes_played']) * 90
                st.metric("Buts/90min", f"{goals_per_90:.2f}")
        
        with stats_cols[3]:
            st.metric("Cartons jaunes", f"{player_data['yellow_cards']:.0f}")
            st.metric("Cartons rouges", f"{player_data['red_cards']:.0f}")
    
    with tab4:
        st.header("🤖 Prédictions et Analyses IA")
        
        # Sélection du joueur pour l'analyse IA
        selected_player_ai = st.selectbox(
            "Choisir un joueur pour l'analyse IA",
            sorted(filtered_df['name'].unique()),
            key="ai_player"
        )
        
        player_ai_data = filtered_df[filtered_df['name'] == selected_player_ai].iloc[0]
        
        # Prédiction du potentiel
        features_for_prediction = np.array([
            player_ai_data['age'],
            player_ai_data['performance_index'],
            player_ai_data['ground_defence'],
            player_ai_data['aerial_play'],
            player_ai_data['distribution'],
            player_ai_data['chance_creation'],
            player_ai_data['take_on'],
            player_ai_data['finishing'],
            player_ai_data['minutes_played'],
            player_ai_data['goals'],
            player_ai_data['assists']
        ])
        
        potential_score = st.session_state.ml_model.predict_potential(features_for_prediction)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-container">
                <h3>🎯 Score de Potentiel IA</h3>
                <h1>{potential_score:.1f}/100</h1>
                <p>Ce score combine l'âge, les performances actuelles et les statistiques pour prédire le potentiel futur du joueur.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Interprétation du score
            if potential_score >= 80:
                st.success("🌟 Potentiel exceptionnel - Talent mondial")
            elif potential_score >= 70:
                st.info("⭐ Très bon potentiel - Joueur prometteur")
            elif potential_score >= 60:
                st.warning("💫 Potentiel intéressant - À surveiller")
            else:
                st.error("📊 Potentiel limité - Développement nécessaire")
        
        with col2:
            # Facteurs d'influence
            st.subheader("Facteurs d'influence")
            
            age_factor = max(0, (21 - player_ai_data['age']) / 4) * 30
            performance_factor = player_ai_data['performance_index'] / 100 * 30
            
            factors_df = pd.DataFrame({
                'Facteur': ['Âge', 'Performance', 'Expérience', 'Statistiques'],
                'Impact': [age_factor, performance_factor, 
                          min(30, player_ai_data['minutes_played']/100), 
                          min(10, (player_ai_data['goals'] + player_ai_data['assists'])/2)]
            })
            
            fig_factors = px.bar(
                factors_df,
                x='Facteur',
                y='Impact',
                title="Facteurs influençant le potentiel",
                color='Impact',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_factors, use_container_width=True)
        
        # Joueurs similaires
        st.subheader("🔍 Joueurs similaires")
        
        similar_players = st.session_state.ml_model.get_similar_players(
            features_for_prediction,
            filtered_df,
            n_similar=5
        )
        
        if not similar_players.empty:
            for idx, (_, similar_player) in enumerate(similar_players.iterrows(), 1):
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <strong>{idx}. {similar_player['name']}</strong> - {similar_player['club']} ({similar_player['league']})
                    <br>Performance: {similar_player['performance_index']:.1f}, 
                    Valeur: €{similar_player['transfer_value_avg']:.1f}M,
                    Buts: {similar_player['goals']:.0f}, Passes: {similar_player['assists']:.0f}
                </div>
                """, unsafe_allow_html=True)
        
        # Recommandations
        st.subheader("💡 Recommandations")
        
        recommendations = []
        
        if player_ai_data['age'] < 19:
            recommendations.append("🎯 Joueur très jeune - Potentiel de développement élevé")
        
        if player_ai_data['performance_index'] > 80:
            recommendations.append("⚡ Performances exceptionnelles - Prêt pour le haut niveau")
        
        if player_ai_data['transfer_value_avg'] < 10:
            recommendations.append("💰 Excellent rapport qualité-prix")
        
        if player_ai_data['minutes_played'] > 800:
            recommendations.append("🏃 Joueur expérimenté avec du temps de jeu significatif")
        
        if player_ai_data['goals'] > 5:
            recommendations.append("⚽ Excellente efficacité offensive")
        
        if player_ai_data['assists'] > 5:
            recommendations.append("🎯 Très bon créateur de jeu")
        
        for rec in recommendations:
            st.success(rec)

if __name__ == "__main__":
    main()
