import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="CIES Football Observatory - U21 Scouting Report",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 0.5rem 0;
    }
    .player-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger et préparer les données
@st.cache_data
def load_data():
    """Charge les données des joueurs depuis le CSV"""
    try:
        # Tentative de chargement du fichier CSV réel
        df = pd.read_csv('Joueurs_2004.csv')
        return df
    except FileNotFoundError:
        # Génération de données simulées si le fichier n'existe pas
        st.warning("Fichier Joueurs_2004.csv non trouvé. Utilisation de données simulées.")
        return generate_sample_data()

def generate_sample_data():
    """Génère des données simulées basées sur le rapport CIES"""
    np.random.seed(42)
    
    # Noms de joueurs fictifs
    first_names = ['Lamine', 'Samu', 'Warren', 'João', 'Pau', 'Kendry', 'Kenan', 'Alejandro', 
                   'Rico', 'Kobbie', 'Endrick', 'Jude', 'Jamal', 'Eduardo', 'Gavi']
    last_names = ['Yamal', 'Aghehowa', 'Zaïre-Emery', 'Neves', 'Cubarsí', 'Páez', 'Yildiz',
                  'Garnacho', 'Lewis', 'Mainoo', 'Felipe', 'Bellingham', 'Musiala', 'Camavinga', 'Gavi']
    
    # Clubs et ligues
    clubs = ['FC Barcelona', 'Real Madrid', 'Manchester City', 'Paris St-Germain', 'Bayern München',
             'Borussia Dortmund', 'Juventus FC', 'AC Milan', 'Arsenal FC', 'Chelsea FC']
    leagues = ['ESP La Liga', 'ENG Premier League', 'GER Bundesliga', 'FRA Ligue 1', 'ITA Serie A']
    
    # Positions selon le rapport CIES
    positions = ['GK', 'CB', 'LB', 'RB', 'DM', 'CM', 'AM', 'LW', 'RW', 'CF']
    
    # Catégories détaillées
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
    
    n_players = 500
    
    data = {
        'Name': [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_players)],
        'Age': np.random.uniform(17.0, 20.9, n_players),
        'Club': np.random.choice(clubs, n_players),
        'League': np.random.choice(leagues, n_players),
        'Position': np.random.choice(positions, n_players),
        'Category': np.random.choice(categories, n_players),
        'Performance_Score': np.random.uniform(50, 95, n_players),
        'Transfer_Value_Min': np.random.uniform(1, 50, n_players),
        'Transfer_Value_Max': np.random.uniform(5, 100, n_players),
        'Ground_Defence': np.random.uniform(10, 95, n_players),
        'Aerial_Play': np.random.uniform(10, 95, n_players),
        'Distribution': np.random.uniform(10, 95, n_players),
        'Chance_Creation': np.random.uniform(10, 95, n_players),
        'Take_On': np.random.uniform(10, 95, n_players),
        'Finishing': np.random.uniform(10, 95, n_players),
        'Minutes_Played': np.random.randint(500, 3000, n_players),
        'Goals': np.random.randint(0, 25, n_players),
        'Assists': np.random.randint(0, 20, n_players),
        'Contract_End': np.random.choice(['2025', '2026', '2027', '2028', '2029'], n_players)
    }
    
    df = pd.DataFrame(data)
    
    # Ajustements logiques
    df.loc[df['Position'] == 'GK', ['Chance_Creation', 'Take_On', 'Finishing']] *= 0.3
    df.loc[df['Position'] == 'CB', ['Chance_Creation', 'Finishing']] *= 0.4
    df.loc[df['Position'].isin(['LW', 'RW', 'CF']), ['Ground_Defence', 'Aerial_Play']] *= 0.4
    
    # Assurer la cohérence Transfer_Value_Max > Transfer_Value_Min
    df['Transfer_Value_Max'] = np.maximum(df['Transfer_Value_Min'] * 1.2, df['Transfer_Value_Max'])
    
    return df

class U21ScoutingApp:
    def __init__(self):
        self.df = load_data()
        self.scaler = StandardScaler()
        self.ml_model = None
        
    def train_ml_model(self):
        """Entraîne le modèle ML pour le score de potentiel"""
        features = ['Age', 'Performance_Score', 'Ground_Defence', 'Aerial_Play', 
                   'Distribution', 'Chance_Creation', 'Take_On', 'Finishing', 'Minutes_Played']
        
        X = self.df[features].fillna(0)
        
        # Score de potentiel basé sur performance et âge (plus jeune = plus de potentiel)
        age_factor = (21 - self.df['Age']) / 4  # Plus l'âge est jeune, plus le facteur est élevé
        performance_factor = self.df['Performance_Score'] / 100
        y = (performance_factor * 0.7 + age_factor * 0.3) * 100
        
        self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.ml_model.fit(X, y)
        
        # Prédictions de potentiel
        self.df['ML_Potential_Score'] = self.ml_model.predict(X)
        
        return self.ml_model.feature_importances_
    
    def calculate_player_similarity(self, player_idx, top_n=5):
        """Calcule la similarité entre joueurs"""
        features = ['Performance_Score', 'Ground_Defence', 'Aerial_Play', 
                   'Distribution', 'Chance_Creation', 'Take_On', 'Finishing']
        
        player_features = self.df.loc[player_idx, features].values.reshape(1, -1)
        all_features = self.df[features].values
        
        similarity_scores = cosine_similarity(player_features, all_features)[0]
        similar_indices = np.argsort(similarity_scores)[::-1][1:top_n+1]  # Exclure le joueur lui-même
        
        return similar_indices, similarity_scores[similar_indices]
    
    def create_radar_chart(self, player_data):
        """Crée un graphique radar pour un joueur"""
        categories = ['Ground Defence', 'Aerial Play', 'Distribution', 
                     'Chance Creation', 'Take On', 'Finishing']
        
        values = [player_data['Ground_Defence'], player_data['Aerial_Play'],
                 player_data['Distribution'], player_data['Chance_Creation'],
                 player_data['Take_On'], player_data['Finishing']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=player_data['Name'],
            line_color='rgb(42, 82, 152)',
            fillcolor='rgba(42, 82, 152, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"Profil de performance - {player_data['Name']}",
            height=500
        )
        
        return fig
    
    def main_dashboard(self):
        """Interface principale du dashboard"""
        
        # En-tête
        st.markdown("""
        <div class="main-header">
            <h1>🏆 CIES Football Observatory</h1>
            <h2>Scouting Report - Meilleurs Joueurs U21 Mondial</h2>
            <p>Rapport interactif basé sur l'analyse des performances et du machine learning</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar pour les filtres
        with st.sidebar:
            st.header("🔍 Filtres de Recherche")
            
            # Filtres
            selected_leagues = st.multiselect(
                "Ligues",
                options=sorted(self.df['League'].unique()),
                default=sorted(self.df['League'].unique())[:3]
            )
            
            selected_positions = st.multiselect(
                "Positions",
                options=sorted(self.df['Position'].unique()),
                default=sorted(self.df['Position'].unique())
            )
            
            age_range = st.slider(
                "Âge",
                min_value=float(self.df['Age'].min()),
                max_value=float(self.df['Age'].max()),
                value=(17.0, 20.9),
                step=0.1
            )
            
            min_performance = st.slider(
                "Score de Performance Minimum",
                min_value=int(self.df['Performance_Score'].min()),
                max_value=int(self.df['Performance_Score'].max()),
                value=60
            )
        
        # Filtrage des données
        filtered_df = self.df[
            (self.df['League'].isin(selected_leagues)) &
            (self.df['Position'].isin(selected_positions)) &
            (self.df['Age'] >= age_range[0]) &
            (self.df['Age'] <= age_range[1]) &
            (self.df['Performance_Score'] >= min_performance)
        ].copy()
        
        # Entraînement du modèle ML
        if self.ml_model is None:
            with st.spinner("🤖 Entraînement du modèle de Machine Learning..."):
                feature_importance = self.train_ml_model()
        
        # Onglets principaux
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vue d'ensemble", 
            "🏅 Classements", 
            "👤 Profils Joueurs", 
            "🤖 Insights ML",
            "📈 Analyses Avancées"
        ])
        
        with tab1:
            self.overview_tab(filtered_df)
        
        with tab2:
            self.rankings_tab(filtered_df)
        
        with tab3:
            self.player_profiles_tab(filtered_df)
        
        with tab4:
            self.ml_insights_tab(filtered_df)
        
        with tab5:
            self.advanced_analysis_tab(filtered_df)
    
    def overview_tab(self, df):
        """Onglet vue d'ensemble"""
        st.header("📊 Vue d'ensemble des Talents U21")
        
        # Métriques clés
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Joueurs", len(df))
        
        with col2:
            avg_age = df['Age'].mean()
            st.metric("Âge Moyen", f"{avg_age:.1f} ans")
        
        with col3:
            avg_performance = df['Performance_Score'].mean()
            st.metric("Performance Moyenne", f"{avg_performance:.1f}/100")
        
        with col4:
            avg_value = df['Transfer_Value_Min'].mean()
            st.metric("Valeur Moyenne", f"€{avg_value:.1f}M")
        
        # Graphiques de distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pos = px.bar(
                df['Position'].value_counts().reset_index(),
                x='index', y='Position',
                title="Distribution par Position"
            )
            st.plotly_chart(fig_pos, use_container_width=True)
        
        with col2:
            fig_league = px.pie(
                df, names='League',
                title="Répartition par Ligue"
            )
            st.plotly_chart(fig_league, use_container_width=True)
    
    def rankings_tab(self, df):
        """Onglet classements"""
        st.header("🏅 Classements par Catégorie")
        
        # Sélection de catégorie
        category = st.selectbox(
            "Choisir une catégorie",
            options=sorted(df['Category'].unique())
        )
        
        category_df = df[df['Category'] == category].sort_values(
            'Performance_Score', ascending=False
        ).head(20)
        
        if not category_df.empty:
            st.subheader(f"Top 20 - {category}")
            
            # Tableau de classement
            display_df = category_df[[
                'Name', 'Age', 'Club', 'Performance_Score', 
                'Transfer_Value_Min', 'Transfer_Value_Max', 'ML_Potential_Score'
            ]].copy()
            
            display_df['Age'] = display_df['Age'].round(1)
            display_df['Performance_Score'] = display_df['Performance_Score'].round(1)
            display_df['ML_Potential_Score'] = display_df['ML_Potential_Score'].round(1)
            display_df['Transfer_Value_Min'] = display_df['Transfer_Value_Min'].round(1)
            display_df['Transfer_Value_Max'] = display_df['Transfer_Value_Max'].round(1)
            
            st.dataframe(
                display_df,
                column_config={
                    "Name": "Nom",
                    "Age": "Âge",
                    "Club": "Club",
                    "Performance_Score": st.column_config.NumberColumn(
                        "Score Performance",
                        format="%.1f/100"
                    ),
                    "ML_Potential_Score": st.column_config.NumberColumn(
                        "Potentiel ML",
                        format="%.1f/100"
                    ),
                    "Transfer_Value_Min": st.column_config.NumberColumn(
                        "Valeur Min (€M)",
                        format="€%.1fM"
                    ),
                    "Transfer_Value_Max": st.column_config.NumberColumn(
                        "Valeur Max (€M)",
                        format="€%.1fM"
                    )
                },
                use_container_width=True
            )
        else:
            st.warning("Aucun joueur trouvé pour cette catégorie avec les filtres actuels.")
    
    def player_profiles_tab(self, df):
        """Onglet profils des joueurs"""
        st.header("👤 Profils Détaillés des Joueurs")
        
        # Sélection d'un joueur
        player_name = st.selectbox(
            "Choisir un joueur",
            options=sorted(df['Name'].unique())
        )
        
        player_data = df[df['Name'] == player_name].iloc[0]
        
        # Informations du joueur
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
            <div class="player-card">
                <h3>{player_data['Name']}</h3>
                <p><strong>Âge:</strong> {player_data['Age']:.1f} ans</p>
                <p><strong>Club:</strong> {player_data['Club']}</p>
                <p><strong>Position:</strong> {player_data['Position']}</p>
                <p><strong>Catégorie:</strong> {player_data['Category']}</p>
                <p><strong>Score Performance:</strong> {player_data['Performance_Score']:.1f}/100</p>
                <p><strong>Potentiel ML:</strong> {player_data['ML_Potential_Score']:.1f}/100</p>
                <p><strong>Valeur Estimée:</strong> €{player_data['Transfer_Value_Min']:.1f}M - €{player_data['Transfer_Value_Max']:.1f}M</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Graphique radar
            radar_fig = self.create_radar_chart(player_data)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        # Joueurs similaires
        st.subheader("🔍 Joueurs Similaires")
        player_idx = df[df['Name'] == player_name].index[0]
        similar_indices, similarity_scores = self.calculate_player_similarity(player_idx)
        
        similar_df = df.iloc[similar_indices][['Name', 'Club', 'Position', 'Performance_Score', 'Transfer_Value_Min']].copy()
        similar_df['Similarity'] = similarity_scores
        similar_df['Similarity'] = (similar_df['Similarity'] * 100).round(1)
        
        st.dataframe(
            similar_df,
            column_config={
                "Similarity": st.column_config.NumberColumn(
                    "Similarité (%)",
                    format="%.1f%%"
                )
            },
            use_container_width=True
        )
    
    def ml_insights_tab(self, df):
        """Onglet insights ML"""
        st.header("🤖 Insights Machine Learning")
        
        # Explication de la métrique ML
        st.markdown("""
        ### 💡 Score de Potentiel ML
        Notre algorithme de machine learning calcule un **Score de Potentiel** qui combine :
        - **Performance actuelle** (70%) : Score technique basé sur 6 domaines d'analyse
        - **Facteur âge** (30%) : Plus le joueur est jeune, plus son potentiel de progression est élevé
        
        Cette métrique permet d'identifier les talents avec le meilleur potentiel de développement.
        """)
        
        # Top potentiels
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚀 Top 10 Potentiels ML")
            top_potential = df.nlargest(10, 'ML_Potential_Score')[
                ['Name', 'Age', 'Club', 'ML_Potential_Score', 'Performance_Score']
            ]
            st.dataframe(top_potential, use_container_width=True)
        
        with col2:
            st.subheader("📈 Graphique Potentiel vs Performance")
            fig_scatter = px.scatter(
                df, 
                x='Performance_Score', 
                y='ML_Potential_Score',
                color='Position',
                hover_data=['Name', 'Age', 'Club'],
                title="Relation Performance vs Potentiel ML"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Analyse des caractéristiques importantes
        if hasattr(self, 'ml_model') and self.ml_model is not None:
            st.subheader("🎯 Importance des Caractéristiques")
            features = ['Age', 'Performance_Score', 'Ground_Defence', 'Aerial_Play', 
                       'Distribution', 'Chance_Creation', 'Take_On', 'Finishing', 'Minutes_Played']
            importance = self.ml_model.feature_importances_
            
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Importance des caractéristiques dans le modèle ML"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    def advanced_analysis_tab(self, df):
        """Onglet analyses avancées"""
        st.header("📈 Analyses Avancées")
        
        # Clustering des joueurs
        st.subheader("🎯 Segmentation des Joueurs par IA")
        
        n_clusters = st.slider("Nombre de groupes", min_value=3, max_value=8, value=5)
        
        # Préparation des données pour clustering
        features_clustering = ['Performance_Score', 'Age', 'Ground_Defence', 'Aerial_Play', 
                              'Distribution', 'Chance_Creation', 'Take_On', 'Finishing']
        
        X_cluster = df[features_clustering].fillna(0)
        
        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df_cluster = df.copy()
        df_cluster['Cluster'] = kmeans.fit_predict(X_cluster)
        
        # PCA pour visualisation
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_cluster)
        
        fig_cluster = px.scatter(
            x=X_pca[:, 0],
            y=X_pca[:, 1],
            color=df_cluster['Cluster'].astype(str),
            hover_data=[df_cluster['Name'], df_cluster['Position'], df_cluster['Performance_Score']],
            title="Segmentation des joueurs (visualisation PCA)"
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Analyse par cluster
        st.subheader("📊 Caractéristiques des Groupes")
        
        cluster_analysis = df_cluster.groupby('Cluster').agg({
            'Age': 'mean',
            'Performance_Score': 'mean',
            'Transfer_Value_Min': 'mean',
            'ML_Potential_Score': 'mean'
        }).round(2)
        
        st.dataframe(cluster_analysis, use_container_width=True)
        
        # Analyse des tendances par âge
        st.subheader("📈 Tendances par Âge")
        
        age_analysis = df.groupby(df['Age'].round()).agg({
            'Performance_Score': 'mean',
            'ML_Potential_Score': 'mean',
            'Transfer_Value_Min': 'mean'
        }).reset_index()
        
        fig_age = px.line(
            age_analysis,
            x='Age',
            y=['Performance_Score', 'ML_Potential_Score'],
            title="Évolution Performance et Potentiel par Âge"
        )
        st.plotly_chart(fig_age, use_container_width=True)

# Point d'entrée principal
def main():
    app = U21ScoutingApp()
    app.main_dashboard()

if __name__ == "__main__":
    main()
