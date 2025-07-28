import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
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
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2a5298;
        margin: 10px 0;
    }
    .player-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .upload-section {
        background: #e8f4f8;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        border: 2px dashed #2a5298;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour charger et traiter les données
@st.cache_data
def load_custom_data(uploaded_file):
    """
    Charge les données depuis un fichier uploadé et les standardise
    """
    try:
        # Déterminer le type de fichier et le charger
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non supporté. Veuillez utiliser CSV ou Excel.")
            return None
        
        # Afficher un aperçu des colonnes pour le mapping
        st.write("**Aperçu des données chargées:**")
        st.write(f"Nombre de lignes: {len(df)}")
        st.write(f"Colonnes disponibles: {list(df.columns)}")
        st.dataframe(df.head())
        
        return df
    
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier: {str(e)}")
        return None

def standardize_data(df, column_mapping):
    """
    Standardise les données selon le format attendu
    """
    try:
        # Créer un nouveau DataFrame avec les colonnes standardisées
        standardized_df = pd.DataFrame()
        
        # Mapping des colonnes obligatoires
        required_columns = {
            'name': 'Nom du joueur',
            'age': 'Âge',
            'club': 'Club',
            'league': 'Championnat',
            'position': 'Position/Catégorie'
        }
        
        # Mapping des colonnes de performance (optionnelles)
        performance_columns = {
            'ground_defence': 'Défense au sol',
            'aerial_play': 'Jeu aérien',
            'distribution': 'Distribution',
            'chance_creation': 'Création de chances',
            'take_on': 'Dribbles/Take-on',
            'finishing': 'Finition',
            'performance_index': 'Index de performance',
            'transfer_value': 'Valeur de transfert',
            'minutes_played': 'Minutes jouées',
            'goals': 'Buts',
            'assists': 'Passes décisives'
        }
        
        # Application du mapping
        for std_col, original_col in column_mapping.items():
            if original_col and original_col in df.columns:
                standardized_df[std_col] = df[original_col]
        
        # Conversion des types de données
        numeric_columns = ['age', 'ground_defence', 'aerial_play', 'distribution', 
                          'chance_creation', 'take_on', 'finishing', 'performance_index',
                          'transfer_value', 'minutes_played', 'goals', 'assists']
        
        for col in numeric_columns:
            if col in standardized_df.columns:
                standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        
        # Calcul de l'index de performance s'il n'existe pas
        if 'performance_index' not in standardized_df.columns:
            performance_cols = ['ground_defence', 'aerial_play', 'distribution', 
                              'chance_creation', 'take_on', 'finishing']
            available_perf_cols = [col for col in performance_cols if col in standardized_df.columns]
            
            if available_perf_cols:
                standardized_df['performance_index'] = standardized_df[available_perf_cols].mean(axis=1)
        
        # Calcul de la valeur de transfert estimée si elle n'existe pas
        if 'transfer_value' not in standardized_df.columns and 'performance_index' in standardized_df.columns:
            # Estimation basique basée sur l'âge et la performance
            age_factor = (21 - standardized_df['age'].fillna(20)) / 4
            perf_factor = standardized_df['performance_index'].fillna(50) / 100
            standardized_df['transfer_value'] = (age_factor * 20 + perf_factor * 30).clip(1, 100)
        
        # Remplissage des valeurs manquantes
        standardized_df['country'] = standardized_df.get('country', 'Unknown')
        standardized_df['minutes_played'] = standardized_df.get('minutes_played', 1000)
        standardized_df['goals'] = standardized_df.get('goals', 0)
        standardized_df['assists'] = standardized_df.get('assists', 0)
        
        # Filtrer les joueurs U21
        if 'age' in standardized_df.columns:
            standardized_df = standardized_df[standardized_df['age'] <= 21]
        
        return standardized_df
    
    except Exception as e:
        st.error(f"Erreur lors de la standardisation: {str(e)}")
        return None

# Classe pour le modèle de Machine Learning (identique à la version précédente)
class PlayerPotentialModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.is_fitted = False
        self.feature_columns = []
    
    def train(self, df):
        # Définir les colonnes de features disponibles
        potential_features = ['age', 'performance_index', 'ground_defence', 'aerial_play', 
                            'distribution', 'chance_creation', 'take_on', 'finishing',
                            'minutes_played', 'goals', 'assists']
        
        # Sélectionner seulement les features disponibles dans les données
        self.feature_columns = [col for col in potential_features if col in df.columns]
        
        if len(self.feature_columns) < 3:
            st.warning("Pas assez de features numériques pour l'entraînement du modèle ML")
            return self
        
        X = df[self.feature_columns].copy()
        
        # Traitement des valeurs manquantes
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Target: utiliser transfer_value ou performance_index
        if 'transfer_value' in df.columns:
            y = df['transfer_value'].fillna(df['transfer_value'].mean())
        else:
            y = df['performance_index'].fillna(df['performance_index'].mean())
        
        self.model.fit(X_scaled, y)
        self.kmeans.fit(X_scaled)
        self.is_fitted = True
        
        return self
    
    def predict_potential(self, player_data):
        if not self.is_fitted or len(self.feature_columns) < 3:
            return 50  # Score par défaut
        
        try:
            # Créer un array avec les features dans le bon ordre
            feature_values = []
            for col in self.feature_columns:
                feature_values.append(player_data.get(col, 0))
            
            X = np.array(feature_values).reshape(1, -1)
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            
            predicted_value = self.model.predict(X_scaled)[0]
            
            # Calcul du potentiel
            age = player_data.get('age', 20)
            performance = player_data.get('performance_index', 50)
            
            age_factor = max(0, (21 - age) / 4) * 30
            performance_factor = performance / 100 * 30
            
            potential = (predicted_value * 0.4 + age_factor + performance_factor)
            return min(100, max(0, potential))
        
        except Exception as e:
            st.warning(f"Erreur dans la prédiction: {str(e)}")
            return 50
    
    def get_similar_players(self, player_data, df, n_similar=5):
        if not self.is_fitted or len(df) < n_similar:
            return pd.DataFrame()
        
        try:
            X = df[self.feature_columns].copy()
            X_imputed = self.imputer.transform(X)
            X_scaled = self.scaler.transform(X_imputed)
            
            # Données du joueur de référence
            feature_values = []
            for col in self.feature_columns:
                feature_values.append(player_data.get(col, 0))
            
            player_X = np.array(feature_values).reshape(1, -1)
            player_X_imputed = self.imputer.transform(player_X)
            player_scaled = self.scaler.transform(player_X_imputed)
            
            # Calcul des distances
            distances = np.sum((X_scaled - player_scaled) ** 2, axis=1)
            similar_indices = np.argsort(distances)[1:n_similar+1]
            
            return df.iloc[similar_indices]
        
        except Exception as e:
            st.warning(f"Erreur dans la recherche de joueurs similaires: {str(e)}")
            return pd.DataFrame()

# Fonction pour créer un radar chart
def create_radar_chart(player_data, player_name):
    performance_columns = ['ground_defence', 'aerial_play', 'distribution', 
                          'chance_creation', 'take_on', 'finishing']
    
    categories = ['Défense au sol', 'Jeu aérien', 'Distribution', 
                  'Création de chances', 'Dribbles', 'Finition']
    
    values = []
    available_categories = []
    
    for i, col in enumerate(performance_columns):
        if col in player_data and pd.notna(player_data[col]):
            values.append(player_data[col])
            available_categories.append(categories[i])
    
    if not values:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=available_categories,
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
        <h1>⚽ SCOUTING REPORT PERSONNALISÉ - JOUEURS U21</h1>
        <p>Analyse des performances et potentiel basée sur vos données</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section d'upload
    st.markdown("""
    <div class="upload-section">
        <h3>📁 Chargement des données</h3>
        <p>Uploadez votre fichier contenant les données des joueurs (CSV ou Excel)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisir un fichier",
        type=['csv', 'xlsx', 'xls'],
        help="Formats supportés: CSV, Excel (.xlsx, .xls)"
    )
    
    if uploaded_file is not None:
        # Charger les données
        raw_df = load_custom_data(uploaded_file)
        
        if raw_df is not None:
            st.success(f"✅ Fichier chargé avec succès! {len(raw_df)} lignes détectées.")
            
            # Interface de mapping des colonnes
            st.subheader("🔗 Mapping des colonnes")
            st.write("Associez les colonnes de votre fichier aux champs standardisés:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Colonnes obligatoires:**")
                name_col = st.selectbox("Nom du joueur", [''] + list(raw_df.columns))
                age_col = st.selectbox("Âge", [''] + list(raw_df.columns))
                club_col = st.selectbox("Club", [''] + list(raw_df.columns))
                league_col = st.selectbox("Championnat", [''] + list(raw_df.columns))
                position_col = st.selectbox("Position/Catégorie", [''] + list(raw_df.columns))
            
            with col2:
                st.write("**Colonnes de performance (optionnelles):**")
                ground_defence_col = st.selectbox("Défense au sol", [''] + list(raw_df.columns))
                aerial_play_col = st.selectbox("Jeu aérien", [''] + list(raw_df.columns))
                distribution_col = st.selectbox("Distribution", [''] + list(raw_df.columns))
                chance_creation_col = st.selectbox("Création de chances", [''] + list(raw_df.columns))
                take_on_col = st.selectbox("Dribbles/Take-on", [''] + list(raw_df.columns))
                finishing_col = st.selectbox("Finition", [''] + list(raw_df.columns))
            
            # Colonnes additionnelles
            with col1:
                st.write("**Autres colonnes:**")
                performance_index_col = st.selectbox("Index de performance", [''] + list(raw_df.columns))
                transfer_value_col = st.selectbox("Valeur de transfert", [''] + list(raw_df.columns))
                country_col = st.selectbox("Pays", [''] + list(raw_df.columns))
            
            with col2:
                st.write("**Statistiques:**")
                minutes_col = st.selectbox("Minutes jouées", [''] + list(raw_df.columns))
                goals_col = st.selectbox("Buts", [''] + list(raw_df.columns))
                assists_col = st.selectbox("Passes décisives", [''] + list(raw_df.columns))
            
            # Bouton pour traiter les données
            if st.button("🚀 Traiter les données", type="primary"):
                # Créer le mapping
                column_mapping = {
                    'name': name_col,
                    'age': age_col,
                    'club': club_col,
                    'league': league_col,
                    'position': position_col,
                    'ground_defence': ground_defence_col,
                    'aerial_play': aerial_play_col,
                    'distribution': distribution_col,
                    'chance_creation': chance_creation_col,
                    'take_on': take_on_col,
                    'finishing': finishing_col,
                    'performance_index': performance_index_col,
                    'transfer_value': transfer_value_col,
                    'country': country_col,
                    'minutes_played': minutes_col,
                    'goals': goals_col,
                    'assists': assists_col
                }
                
                # Standardiser les données
                df = standardize_data(raw_df, column_mapping)
                
                if df is not None and len(df) > 0:
                    st.success(f"✅ Données traitées! {len(df)} joueurs U21 identifiés.")
                    
                    # Stocker les données dans la session
                    st.session_state.processed_data = df
                    st.session_state.data_processed = True
                    
                    # Initialiser le modèle ML
                    st.session_state.ml_model = PlayerPotentialModel().train(df)
                else:
                    st.error("❌ Erreur lors du traitement des données.")
    
    # Interface principale (seulement si les données sont traitées)
    if hasattr(st.session_state, 'data_processed') and st.session_state.data_processed:
        df = st.session_state.processed_data
        
        # Sidebar pour les filtres
        st.sidebar.header("🔍 Filtres de recherche")
        
        # Filtres dynamiques basés sur les données
        available_positions = df['position'].dropna().unique() if 'position' in df.columns else []
        available_leagues = df['league'].dropna().unique() if 'league' in df.columns else []
        
        selected_position = st.sidebar.selectbox(
            "Position/Catégorie",
            ['Toutes'] + list(available_positions)
        )
        
        selected_league = st.sidebar.selectbox(
            "Championnat",
            ['Tous'] + list(available_leagues)
        )
        
        if 'age' in df.columns:
            min_age, max_age = int(df['age'].min()), int(df['age'].max())
            age_range = st.sidebar.slider(
                "Âge",
                min_value=min_age,
                max_value=max_age,
                value=(min_age, max_age)
            )
        
        if 'performance_index' in df.columns:
            min_perf, max_perf = int(df['performance_index'].min()), int(df['performance_index'].max())
            min_performance = st.sidebar.slider(
                "Performance minimale",
                min_value=min_perf,
                max_value=max_perf,
                value=min_perf
            )
        
        # Application des filtres
        filtered_df = df.copy()
        
        if selected_position != 'Toutes' and 'position' in df.columns:
            filtered_df = filtered_df[filtered_df['position'] == selected_position]
        
        if selected_league != 'Tous' and 'league' in df.columns:
            filtered_df = filtered_df[filtered_df['league'] == selected_league]
        
        if 'age' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['age'] >= age_range[0]) & 
                (filtered_df['age'] <= age_range[1])
            ]
        
        if 'performance_index' in df.columns:
            filtered_df = filtered_df[filtered_df['performance_index'] >= min_performance]
        
        # Métriques globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Joueurs analysés", len(filtered_df))
        
        with col2:
            if 'performance_index' in filtered_df.columns:
                avg_performance = filtered_df['performance_index'].mean()
                st.metric("Performance moyenne", f"{avg_performance:.1f}")
            else:
                st.metric("Performance moyenne", "N/A")
        
        with col3:
            if 'transfer_value' in filtered_df.columns:
                avg_value = filtered_df['transfer_value'].mean()
                st.metric("Valeur moyenne", f"€{avg_value:.1f}M")
            else:
                st.metric("Valeur moyenne", "N/A")
        
        with col4:
            if 'age' in filtered_df.columns:
                avg_age = filtered_df['age'].mean()
                st.metric("Âge moyen", f"{avg_age:.1f} ans")
            else:
                st.metric("Âge moyen", "N/A")
        
        # Onglets principaux
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Top Joueurs", 
            "📊 Analyses", 
            "🎯 Joueur Détaillé", 
            "🤖 Prédictions IA"
        ])
        
        with tab1:
            st.header("🏆 Classement des meilleurs joueurs")
            
            # Tri par performance ou par une autre métrique
            sort_column = 'performance_index' if 'performance_index' in filtered_df.columns else filtered_df.select_dtypes(include=[np.number]).columns[0]
            top_players = filtered_df.nlargest(20, sort_column)
            
            for idx, (_, player) in enumerate(top_players.iterrows()):
                with st.container():
                    st.markdown(f"""
                    <div class="player-card">
                        <h3>#{idx + 1} {player.get('name', 'N/A')}</h3>
                        <div style="display: flex; justify-content: space-between;">
                            <div>
                                <p><strong>Club:</strong> {player.get('club', 'N/A')}</p>
                                <p><strong>Âge:</strong> {player.get('age', 'N/A')} ans</p>
                                <p><strong>Position:</strong> {player.get('position', 'N/A')}</p>
                            </div>
                            <div>
                                <p><strong>Performance:</strong> {player.get('performance_index', 'N/A')}</p>
                                <p><strong>Valeur estimée:</strong> €{player.get('transfer_value', 'N/A')}M</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.header("📊 Analyses statistiques")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution par position/catégorie
                if 'position' in filtered_df.columns:
                    position_counts = filtered_df['position'].value_counts()
                    fig_pos = px.bar(
                        x=position_counts.values,
                        y=position_counts.index,
                        orientation='h',
                        title="Distribution par position"
                    )
                    st.plotly_chart(fig_pos, use_container_width=True)
            
            with col2:
                # Graphique de corrélation si possible
                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    x_col = st.selectbox("Axe X", numeric_cols, key="scatter_x")
                    y_col = st.selectbox("Axe Y", numeric_cols, key="scatter_y", index=1)
                    
                    fig_scatter = px.scatter(
                        filtered_df,
                        x=x_col,
                        y=y_col,
                        hover_data=['name'] if 'name' in filtered_df.columns else None,
                        title=f"{x_col} vs {y_col}"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab3:
            st.header("🎯 Analyse détaillée d'un joueur")
            
            if 'name' in filtered_df.columns:
                selected_player_name = st.selectbox(
                    "Choisir un joueur",
                    filtered_df['name'].unique()
                )
                
                player_data = filtered_df[filtered_df['name'] == selected_player_name].iloc[0]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="player-card">
                        <h2>{player_data.get('name', 'N/A')}</h2>
                        <p><strong>Club:</strong> {player_data.get('club', 'N/A')}</p>
                        <p><strong>Âge:</strong> {player_data.get('age', 'N/A')} ans</p>
                        <p><strong>Position:</strong> {player_data.get('position', 'N/A')}</p>
                        <p><strong>Performance:</strong> {player_data.get('performance_index', 'N/A')}</p>
                        <p><strong>Valeur estimée:</strong> €{player_data.get('transfer_value', 'N/A')}M</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Radar chart si les données de performance sont disponibles
                    radar_fig = create_radar_chart(player_data, player_data.get('name', 'Joueur'))
                    if radar_fig:
                        st.plotly_chart(radar_fig, use_container_width=True)
                    else:
                        st.info("Données de performance insuffisantes pour le radar chart")
        
        with tab4:
            st.header("🤖 Prédictions et Analyses IA")
            
            if 'name' in filtered_df.columns:
                selected_player_ai = st.selectbox(
                    "Choisir un joueur pour l'analyse IA",
                    filtered_df['name'].unique(),
                    key="ai_player"
                )
                
                player_ai_data = filtered_df[filtered_df['name'] == selected_player_ai].iloc[0]
                
                # Prédiction du potentiel
                potential_score = st.session_state.ml_model.predict_potential(player_ai_data.to_dict())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3>🎯 Score de Potentiel IA</h3>
                        <h1 style="color: #2a5298;">{potential_score:.1f}/100</h1>
                        <p>Score basé sur l'analyse des données disponibles</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Interprétation du score
                    if potential_score >= 80:
                        st.success("🌟 Potentiel exceptionnel")
                    elif potential_score >= 70:
                        st.info("⭐ Très bon potentiel")
                    elif potential_score >= 60:
                        st.warning("💫 Potentiel intéressant")
                    else:
                        st.error("📊 Potentiel à développer")
                
                with col2:
                    # Joueurs similaires
                    st.subheader("🔍 Joueurs similaires")
                    similar_players = st.session_state.ml_model.get_similar_players(
                        player_ai_data.to_dict(),
                        filtered_df,
                        n_similar=5
                    )
                    
                    if not similar_players.empty:
                        for _, similar_player in similar_players.iterrows():
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px;">
                                <strong>{similar_player.get('name', 'N/A')}</strong> - {similar_player.get('club', 'N/A')} 
                                (Performance: {similar_player.get('performance_index', 'N/A')}, 
                                Valeur: €{similar_player.get('transfer_value', 'N/A')}M)
                            </div>
                            """, unsafe_allow_html