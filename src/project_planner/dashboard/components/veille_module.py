#!/usr/bin/env python3
"""
🤖 MODULE VEILLE IA - Interface Streamlit
Surveillance intelligente et prédictive des projets
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Any, Optional
import sys
import os

# Import du moteur IA
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ml'))
try:
    from ai_veille_engine import AIVeilleEngine, Alerte, Prediction
except ImportError:
    st.error("❌ Impossible d'importer le moteur IA de veille")
    AIVeilleEngine = None

def render_veille_module(project_context: Dict[str, Any]):
    """
    🎯 Module Veille IA Principal
    """
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    ">
        <h1 style="margin: 0; font-size: 2rem;">🤖 Veille IA Intelligence</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Surveillance prédictive et alertes intelligentes</p>
    </div>
    """, unsafe_allow_html=True)
    
    if AIVeilleEngine is None:
        st.error("Module IA indisponible")
        return
    
    # Initialiser le moteur IA en session
    if 'ai_veille_engine' not in st.session_state:
        st.session_state.ai_veille_engine = AIVeilleEngine()
        # Générer données de démo si pas de contexte
        if not project_context.get('has_real_data', False):
            _generer_donnees_demo(st.session_state.ai_veille_engine)
    
    moteur = st.session_state.ai_veille_engine
    
    # Configuration en sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Veille")
        
        refresh_auto = st.checkbox("🔄 Refresh Auto", value=True)
        if refresh_auto:
            refresh_interval = st.selectbox("Intervalle", [30, 60, 300, 600], index=2)
            st.markdown(f"*Refresh toutes les {refresh_interval}s*")
        
        # Seuils personnalisables
        st.markdown("### 🎚️ Seuils Alertes")
        seuil_budget = st.slider("Budget critique %", 0.7, 0.99, 0.85)
        seuil_qualite = st.slider("Qualité min %", 0.3, 0.8, 0.5)
        seuil_risque = st.slider("Risque max %", 0.5, 0.9, 0.7)
        
        # Mettre à jour config moteur
        moteur.config['seuils_alertes']['budget']['critique'] = seuil_budget
        moteur.config['seuils_alertes']['qualite']['critique'] = seuil_qualite
        moteur.config['seuils_alertes']['risque']['critique'] = seuil_risque
    
    # Layout principal en onglets
    tabs = st.tabs(["🚨 Alertes Live", "📈 Prédictions IA", "📊 Santé Projet", "🎛️ Surveillance"])
    
    with tabs[0]:
        render_alertes_live(moteur)
    
    with tabs[1]:
        render_predictions_ia(moteur)
    
    with tabs[2]:
        render_sante_projet(moteur)
    
    with tabs[3]:
        render_surveillance_avancee(moteur, project_context)

def render_alertes_live(moteur: AIVeilleEngine):
    """🚨 Section Alertes Temps Réel"""
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🚨 Alertes Critiques en Temps Réel")
    
    with col2:
        if st.button("🔄 Actualiser", type="primary"):
            st.info("✅ Données actualisées")
    
    with col3:
        if st.button("🧹 Effacer Cache"):
            moteur.alertes_actives = []
            st.success("Cache effacé")
    
    # Générer alertes
    with st.spinner("🤖 Analyse IA en cours..."):
        alertes = moteur.generer_alertes_intelligentes()
    
    if not alertes:
        st.success("✅ Aucune alerte critique - Projet stable")
        return
    
    # Métriques d'alertes
    col1, col2, col3, col4, col5 = st.columns(5)
    stats = _calculer_stats_alertes(alertes)
    
    with col1:
        st.metric("🔥 Total", stats['total'])
    with col2:
        st.metric("🔴 Critiques", stats['critique'], delta=f"+{stats['critique']}" if stats['critique'] > 0 else None)
    with col3:
        st.metric("🟠 Élevées", stats['elevee'])
    with col4:
        st.metric("🟡 Moyennes", stats['moyenne'])  
    with col5:
        st.metric("🔵 Faibles", stats['faible'])
    
    st.markdown("---")
    
    # Affichage alertes par criticité
    for criticite in ['critique', 'elevee', 'moyenne', 'faible']:
        alertes_niveau = [a for a in alertes if a.criticite == criticite]
        if alertes_niveau:
            _afficher_alertes_niveau(alertes_niveau, criticite)

def render_predictions_ia(moteur: AIVeilleEngine):
    """📈 Section Prédictions IA"""
    
    st.markdown("### 📈 Prédictions Multi-Horizons")
    
    # Paramètres prédiction
    col1, col2 = st.columns(2)
    with col1:
        horizons = st.multiselect("Horizons (jours)", [3, 7, 14, 30], default=[7, 14])
    with col2:
        metriques = st.multiselect("Métriques", 
            ['budget_usage', 'completion_rate', 'quality_score', 'risk_score'],
            default=['budget_usage', 'completion_rate'])
    
    if not horizons or not metriques:
        st.info("Sélectionnez horizons et métriques pour voir prédictions")
        return
    
    # Générer prédictions
    with st.spinner("🧠 Calcul prédictions IA..."):
        predictions = moteur.predire_metriques(horizons)
    
    if not predictions:
        st.warning("⚠️ Pas assez de données pour prédictions fiables")
        return
    
    # Graphiques prédictions
    for metrique in metriques:
        if metrique in predictions and predictions[metrique]:
            _afficher_graphique_predictions(metrique, predictions[metrique], moteur)

def render_sante_projet(moteur: AIVeilleEngine):
    """💚 Section Santé Globale"""
    
    st.markdown("### 💚 Score de Santé Projet")
    
    # Score global
    score_sante = moteur.obtenir_score_sante()
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Gauge score
        _afficher_gauge_sante(score_sante['score'], score_sante['status'])
    
    with col2:
        # Détails par composant
        st.markdown("#### 📊 Détail des Composants")
        if 'details' in score_sante:
            for composant, score in score_sante['details'].items():
                couleur = _get_couleur_score(score)
                st.markdown(f"""
                <div style="
                    background: {couleur};
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    margin: 0.5rem 0;
                    color: white;
                    font-weight: bold;
                ">
                    {composant.title()}: {score:.1f}/100
                </div>
                """, unsafe_allow_html=True)
    
    # Tendance historique
    st.markdown("#### 📈 Évolution Santé Projet")
    if len(moteur.historique_metriques) > 0:
        _afficher_tendance_sante(moteur)

def render_surveillance_avancee(moteur: AIVeilleEngine, project_context: Dict):
    """🎛️ Surveillance Avancée"""
    
    st.markdown("### 🎛️ Surveillance Système Avancée")
    
    # Métriques système
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        nb_points = len(moteur.historique_metriques)
        st.metric("📊 Points données", nb_points)
    
    with col2:
        st.metric("🤖 Modèles entrainés", "✅" if moteur.modeles_entraines else "❌")
    
    with col3:
        st.metric("🚨 Alertes actives", len(moteur.alertes_actives))
    
    with col4:
        st.metric("🔄 Dernière MAJ", datetime.now().strftime("%H:%M:%S"))
    
    # Matrice de corrélation
    if len(moteur.historique_metriques) > 5:
        st.markdown("#### 🔗 Corrélations entre Métriques")
        _afficher_matrice_correlation(moteur)
    
    # Détection anomalies
    st.markdown("#### 🎯 Détection d'Anomalies")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Scanner Anomalies", type="secondary"):
            with st.spinner("Analyse en cours..."):
                anomalies = moteur.detecter_anomalies()
                st.session_state.derniere_detection = anomalies
    
    with col2:
        if st.button("📤 Exporter Données"):
            _exporter_donnees_veille(moteur)
    
    # Affichage anomalies
    if hasattr(st.session_state, 'derniere_detection'):
        anomalies = st.session_state.derniere_detection
        if anomalies:
            st.markdown(f"**🎯 {len(anomalies)} anomalies détectées:**")
            for anom in anomalies[:3]:  # Top 3
                st.error(f"⚠️ {anom.message} (confiance: {anom.score_confiance:.1%})")
        else:
            st.success("✅ Aucune anomalie détectée")
    
    # Configuration avancée
    with st.expander("⚙️ Configuration Avancée"):
        st.markdown("**Paramètres ML:**")
        
        col1, col2 = st.columns(2)
        with col1:
            contamination = st.slider("Taux contamination anomalies", 0.05, 0.2, 0.1)
            fenetre_historique = st.number_input("Fenêtre historique (jours)", 30, 365, 90)
        
        with col2:
            min_confiance = st.slider("Confiance minimale alertes", 0.5, 0.95, 0.7)
            nb_estimators = st.number_input("Nb estimateurs RF", 10, 200, 50)
        
        if st.button("💾 Appliquer Configuration"):
            moteur.config.update({
                'fenetre_historique': fenetre_historique,
                'min_confiance_alerte': min_confiance
            })
            # Recréer détecteur anomalies
            moteur.anomaly_detector.contamination = contamination
            st.success("✅ Configuration mise à jour")

# === FONCTIONS UTILITAIRES ===

def _generer_donnees_demo(moteur: AIVeilleEngine):
    """Génère données de démo pour tester le module"""
    import random
    
    for i in range(45):  # 45 jours de données
        timestamp = datetime.now() - timedelta(days=45-i)
        
        # Simulation évolution réaliste
        base_budget = 0.3 + (i / 45) * 0.5 + random.uniform(-0.1, 0.1)
        base_completion = 0.2 + (i / 45) * 0.7 + random.uniform(-0.1, 0.1)
        base_quality = 0.8 + random.uniform(-0.2, 0.1)
        base_risk = 0.3 + random.uniform(-0.1, 0.3)
        
        metriques = {
            'budget_usage': max(0, min(1, base_budget)),
            'completion_rate': max(0, min(1, base_completion)),  
            'quality_score': max(0, min(1, base_quality)),
            'risk_score': max(0, min(1, base_risk))
        }
        
        moteur.ingerer_donnees(metriques, timestamp)

def _calculer_stats_alertes(alertes: List[Alerte]) -> Dict[str, int]:
    """Calcule statistiques des alertes"""
    stats = {'total': len(alertes), 'critique': 0, 'elevee': 0, 'moyenne': 0, 'faible': 0}
    
    for alerte in alertes:
        if alerte.criticite in stats:
            stats[alerte.criticite] += 1
    
    return stats

def _afficher_alertes_niveau(alertes: List[Alerte], criticite: str):
    """Affiche alertes d'un niveau de criticité"""
    
    couleurs = {
        'critique': '#dc3545',
        'elevee': '#fd7e14', 
        'moyenne': '#ffc107',
        'faible': '#17a2b8'
    }
    
    icones = {
        'critique': '🔴',
        'elevee': '🟠',
        'moyenne': '🟡', 
        'faible': '🔵'
    }
    
    st.markdown(f"#### {icones[criticite]} Alertes {criticite.title()}")
    
    for alerte in alertes[:3]:  # Max 3 par niveau
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(90deg, {couleurs[criticite]}22, {couleurs[criticite]}11);
                    border-left: 4px solid {couleurs[criticite]};
                    padding: 1rem;
                    border-radius: 8px;
                    margin-bottom: 0.5rem;
                ">
                    <strong>{alerte.message}</strong><br>
                    <small>💡 {alerte.recommandation}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Confiance", f"{alerte.score_confiance:.1%}")
            
            with col3:
                if alerte.module_cible and st.button(f"🔗 Aller", key=f"nav_{alerte.id}"):
                    st.session_state.current_view_mode = alerte.module_cible
                    st.info("✅ Données actualisées")

def _afficher_graphique_predictions(metrique: str, predictions: List[Prediction], moteur: AIVeilleEngine):
    """Affiche graphique prédictions pour une métrique"""
    
    st.markdown(f"#### 📊 Prédictions: {metrique}")
    
    # Données historiques
    if len(moteur.historique_metriques) > 0:
        df_hist = moteur.historique_metriques.copy()
        
        # Graphique avec historique + prédictions
        fig = go.Figure()
        
        # Ligne historique
        fig.add_trace(go.Scatter(
            x=df_hist['timestamp'],
            y=df_hist[metrique],
            mode='lines+markers',
            name='Historique',
            line=dict(color='blue')
        ))
        
        # Points prédictions
        for pred in predictions:
            date_pred = datetime.now() + timedelta(days=pred.horizon_jours)
            
            couleur = 'green' if pred.tendance == 'amelioration' else 'red' if pred.tendance == 'degradation' else 'orange'
            
            fig.add_trace(go.Scatter(
                x=[date_pred],
                y=[pred.valeur_predite],
                mode='markers',
                name=f'Pred {pred.horizon_jours}j',
                marker=dict(size=10, color=couleur),
                text=f"Confiance: {pred.score_confiance:.1%}"
            ))
        
        fig.update_layout(
            title=f"Évolution {metrique}",
            xaxis_title="Date",
            yaxis_title="Valeur",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Table détaillée
        df_pred = pd.DataFrame([{
            'Horizon': f"{p.horizon_jours}j",
            'Valeur Prédite': f"{p.valeur_predite:.2%}",
            'Tendance': p.tendance,
            'Confiance': f"{p.score_confiance:.1%}"
        } for p in predictions])
        
        st.dataframe(df_pred, use_container_width=True)

def _afficher_gauge_sante(score: float, status: str):
    """Affiche gauge du score de santé"""
    
    couleurs_status = {
        'excellent': 'green',
        'bon': 'lightgreen', 
        'moyen': 'orange',
        'critique': 'red'
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Score Santé Projet"},
        delta = {'reference': 70},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': couleurs_status.get(status, 'gray')},
            'steps': [
                {'range': [0, 40], 'color': "lightgray"},
                {'range': [40, 70], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
    
    # Status
    st.markdown(f"**Statut:** {status.title()}")

def _get_couleur_score(score: float) -> str:
    """Retourne couleur basée sur score"""
    if score >= 80:
        return '#28a745'
    elif score >= 60:
        return '#17a2b8'
    elif score >= 40:
        return '#ffc107'
    else:
        return '#dc3545'

def _afficher_tendance_sante(moteur: AIVeilleEngine):
    """Affiche tendance historique santé"""
    
    df = moteur.historique_metriques.copy()
    if len(df) < 2:
        return
    
    # Calculer score santé historique
    scores_historiques = []
    for _, row in df.iterrows():
        score_budget = max(0, 100 - (row.get('budget_usage', 0.5) * 100))
        score_completion = row.get('completion_rate', 0.5) * 100
        score_quality = row.get('quality_score', 0.7) * 100
        score_risk = max(0, 100 - (row.get('risk_score', 0.3) * 100))
        
        score_global = (score_budget * 0.25 + score_completion * 0.30 + 
                       score_quality * 0.25 + score_risk * 0.20)
        scores_historiques.append(score_global)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=scores_historiques,
        mode='lines+markers',
        name='Score Santé',
        line=dict(color='purple', width=3)
    ))
    
    fig.update_layout(
        title="Évolution Score Santé Projet",
        xaxis_title="Date",
        yaxis_title="Score (/100)",
        height=300
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _afficher_matrice_correlation(moteur: AIVeilleEngine):
    """Affiche matrice corrélation des métriques"""
    
    df = moteur.historique_metriques.copy()
    metriques = ['budget_usage', 'completion_rate', 'quality_score', 'risk_score']
    
    correlation_matrix = df[metriques].corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Matrice de Corrélation des Métriques"
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _exporter_donnees_veille(moteur: AIVeilleEngine):
    """Exporte données de veille"""
    
    try:
        # Préparer données export
        resume = moteur.obtenir_resume_veille()
        
        # Convertir en JSON
        donnees_export = {
            'timestamp_export': datetime.now().isoformat(),
            'resume_veille': resume,
            'historique_metriques': moteur.historique_metriques.to_dict('records') if len(moteur.historique_metriques) > 0 else [],
            'alertes_actives': [a.__dict__ for a in moteur.alertes_actives]
        }
        
        # Bouton téléchargement
        st.download_button(
            label="📥 Télécharger Rapport Veille",
            data=json.dumps(donnees_export, indent=2, default=str),
            file_name=f"rapport_veille_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )
        
        st.success("✅ Données préparées pour export")
        
    except Exception as e:
        st.error(f"❌ Erreur export: {e}")

# Test si exécuté directement
if __name__ == "__main__":
    st.set_page_config(page_title="Test Module Veille IA", layout="wide")
    
    # Context de test
    test_context = {
        'project_name': 'Projet Test',
        'has_real_data': False
    }
    
    render_veille_module(test_context)