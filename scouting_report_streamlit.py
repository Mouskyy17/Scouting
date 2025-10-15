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

# Données simulées basées sur le rapport CIES
@st.cache_data
def load_data():
    np.random.seed(42)
    
    # Catégories de joueurs basées sur le rapport CIES
    categories = [
        'Short-passes Goalkeepers', 'Long-passes Goalkeepers',
        'Full Defence Centre Backs', 'Allrounder Centre Backs', 'Build-up Centre Backs',
        'Defensive Left Full/Wing Backs', 'Attacking Left Full/Wing Backs',
        'Defensive Right Full/Wing Backs', 'Attacking Right Full/Wing Backs',
        'Holding Midfielders', 'Playmaking Midfielders', 'Assisting Midfielders',
        'Infiltrating Midfielders', 'Shooting Midfielders',
        'Infiltrating Left Wingers', 'Assisting/Shooting Left Wingers',
        'Infiltrating Right Wingers', 'Assisting/Shooting Right Wingers',
        'Allrounder Centre Forwards', 'Target Man Centre Forwards'
    ]
    
    # Noms de joueurs réels du rapport
    top_players = [
        'Lamine Yamal', 'Warren Zaïre-Emery', 'João Neves', 'Pau Cubarsí',
        'Kendry Páez', 'Alejandro Garnacho', 'Kenan Yildiz', 'Rico Lewis',
        'Jorrel Hato', 'Sávio Moreira', 'Kobbie Mainoo', 'Endrick Felipe',
        'Malick Diouf', 'Martim Fernandes', 'Geovany Quenda', 'Claudio Echeverri',
        'Carlos Baleba', 'Aleksandar Pavlović', 'Mario Stroeykens', 'Samu Aghehowa'
    ]
    
    clubs = [
        'FC Barcelona', 'Paris St-Germain', 'Manchester United', 'Real Madrid',
        'Manchester City', 'Juventus', 'Ajax', 'Porto', 'Sporting CP',
        'Brighton & Hove', 'Chelsea', 'Arsenal', 'Bayern München'
    ]
    
    leagues = [
        'La Liga', 'Premier League', 'Ligue 1', 'Serie A', 'Bundesliga',
        'Eredivisie', 'Primeira Liga', 'Pro League'
    ]
    
    countries = [
        'Spain', 'France', 'England', 'Germany', 'Brazil', 'Argentina',
        'Portugal', 'Netherlands', 'Italy', 'Belgium', 'Turkey', 'Ecuador'
    ]
    
    n_players = 500
    
    data = {
        'name': np.random.choice(top_players + [f'Player_{i}' for i in range(100)], n_players),
        'age': np.random.uniform(17, 21, n_players),
        'category': np.random.choice(categories, n_players),
        'club': np.random.choice(clubs, n_players),
        'league': np.random.choice(leagues, n_players),
        'country': np.random.choice(countries, n_players),
        'performance_index': np.random.uniform(50, 95, n_players),
        'transfer_value_min': np.random.uniform(1, 50, n_players),
        'transfer_value_max': np.random.uniform(5, 100, n_players),
        'ground_defence': np.random.uniform(0, 100, n_players),
        'aerial_play': np.random.uniform(0, 100, n_players),
        'distribution': np.random.uniform(0, 100, n_players),
        'chance_creation': np.random.uniform(0, 100, n_players),
        'take_on': np.random.uniform(0, 100, n_players),
        'finishing': np.random.uniform(0, 100, n_players),
        'minutes_played': np.random.randint(500, 3000, n_players),
        'goals': np.random.randint(0, 25, n_players),
        'assists': np.random.randint(0, 20, n_players),
        'yellow_cards': np.random.randint(0, 10, n_players),
        'red_cards': np.random.randint(0, 3, n_players)
    }
    
    df = pd.DataFrame(data)
    df['transfer_value_avg'] = (df['transfer_value_min'] + df['transfer_value_max']) / 2
    
    return df

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
        
        # Correction: player_data est déjà un array numpy
        if isinstance(player_data, np.ndarray):
            X = self.scaler.transform(player_data.reshape(1, -1))
        else:
            X = self.scaler.transform(np.array(player_data).reshape(1, -1))
        
        potential_score = self.model.predict(X)[0]
        
        # Calcul du potentiel basé sur l'âge et les performances
        age_factor = max(0, (21 - player_data[0]) / 4)  # Plus jeune = plus de potentiel
        performance_factor = player_data[1] / 100  # Performance actuelle
        
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
        
        # Correction: player_data est déjà un array numpy
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
        <p>Analyse des performances et potentiel des jeunes talents mondiaux</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement des données
    df = load_data()
    
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
        min_value=17,
        max_value=21,
        value=(17, 21)
    )
    
    min_performance = st.sidebar.slider(
        "Performance minimale",
        min_value=50,
        max_value=95,
        value=60
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
        (filtered_df['performance_index'] >= min_performance)
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
        
        for idx, player in top_players.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="player-card">
                    <h3>#{len(top_players) - list(top_players.index).index(idx)} {player['name']}</h3>
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <p><strong>Club:</strong> {player['club']} ({player['league']})</p>
                            <p><strong>Âge:</strong> {player['age']:.1f} ans</p>
                            <p><strong>Catégorie:</strong> {player['category']}</p>
                        </div>
                        <div>
                            <p><strong>Performance:</strong> {player['performance_index']:.1f}/100</p>
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
            fig_cat.update_layout(height=600)
            st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            # Relation performance vs valeur
            fig_scatter = px.scatter(
                filtered_df,
                x='performance_index',
                y='transfer_value_avg',
                color='category',
                size='age',
                hover_data=['name', 'club'],
                title="Performance vs Valeur de transfert"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Analyse par championnat
        st.subheader("Performance par championnat")
        league_performance = filtered_df.groupby('league').agg({
            'performance_index': 'mean',
            'transfer_value_avg': 'mean',
            'name': 'count'
        }).round(2)
        league_performance.columns = ['Performance moyenne', 'Valeur moyenne (€M)', 'Nombre de joueurs']
        st.dataframe(league_performance.sort_values('Performance moyenne', ascending=False))
    
    with tab3:
        st.header("🎯 Analyse détaillée d'un joueur")
        
        # Sélection du joueur
        selected_player_name = st.selectbox(
            "Choisir un joueur",
            filtered_df['name'].unique()
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
                <p><strong>Âge:</strong> {player_data['age']:.1f} ans</p>
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
        
        stats_cols = st.columns(3)
        
        with stats_cols[0]:
            st.metric("Minutes jouées", f"{player_data['minutes_played']:,}")
            st.metric("Buts", player_data['goals'])
        
        with stats_cols[1]:
            st.metric("Passes décisives", player_data['assists'])
            st.metric("Cartons jaunes", player_data['yellow_cards'])
        
        with stats_cols[2]:
            if player_data['minutes_played'] > 0:
                goals_per_90 = (player_data['goals'] / player_data['minutes_played']) * 90
                st.metric("Buts/90min", f"{goals_per_90:.2f}")
            st.metric("Cartons rouges", player_data['red_cards'])
    
    with tab4:
        st.header("🤖 Prédictions et Analyses IA")
        
        # Sélection du joueur pour l'analyse IA
        selected_player_ai = st.selectbox(
            "Choisir un joueur pour l'analyse IA",
            filtered_df['name'].unique(),
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
                title="Facteurs influençant le potentiel"
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
            for idx, similar_player in similar_players.iterrows():
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <strong>{similar_player['name']}</strong> - {similar_player['club']} 
                    (Performance: {similar_player['performance_index']:.1f}, 
                    Valeur: €{similar_player['transfer_value_avg']:.1f}M)
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
        
        if player_ai_data['minutes_played'] > 2000:
            recommendations.append("🏃 Joueur expérimenté avec du temps de jeu")
        
        for rec in recommendations:
            st.success(rec)

if __name__ == "__main__":
    main()
