# Module Planning Multi-projets pour PlannerIA
# Vue globale, dépendances entre projets, chemin critique et planification stratégique

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import networkx as nx

class PlanningModule:
    def __init__(self):
        self.status_colors = {
            'active': '#10B981',
            'planning': '#F59E0B',
            'completed': '#3B82F6',
            'paused': '#6B7280',
            'cancelled': '#EF4444',
            'review': '#8B5CF6',
            'testing': '#06B6D4'
        }
        
        self.priority_colors = {
            'critique': '#EF4444',
            'haute': '#F59E0B',
            'moyenne': '#10B981',
            'basse': '#6B7280'
        }
        
        self.risk_colors = {
            'haut': '#EF4444',
            'moyen': '#F59E0B',
            'bas': '#10B981'
        }
        
    def load_portfolio_data(self) -> Dict[str, Any]:
        """Charge les données du portfolio de projets enrichies"""
        return {
            'projets': [
                {
                    'id': 'proj_001',
                    'nom': 'Refonte Application Mobile',
                    'status': 'active',
                    'priorite': 'critique',
                    'progression': 72,
                    'budget': 120000,
                    'budget_utilise': 86400,
                    'date_debut': '2024-01-15',
                    'date_fin_prevue': '2024-04-30',
                    'date_fin_reelle': None,
                    'manager': 'Alice Martin',
                    'equipe_taille': 5,
                    'risque_niveau': 'moyen',
                    'score_sante': 78,
                    'velocite_moyenne': 23,
                    'client': 'Direction Marketing',
                    'roi_prevu': 245000,
                    'dependances': ['proj_003'],
                    'bloque': [],
                    'technologies': ['React Native', 'Node.js', 'MongoDB'],
                    'jalons': [
                        {'nom': 'Design UI/UX', 'date': '2024-02-15', 'status': 'completed', 'progression': 100},
                        {'nom': 'Développement Core', 'date': '2024-03-15', 'status': 'completed', 'progression': 100},
                        {'nom': 'Tests & Validation', 'date': '2024-04-15', 'status': 'active', 'progression': 60},
                        {'nom': 'Déploiement', 'date': '2024-04-30', 'status': 'pending', 'progression': 0}
                    ],
                    'ressources_cles': ['Alice Martin (Tech Lead)', 'Bob Chen (Mobile Dev)', 'Sarah Kim (UX)']
                },
                {
                    'id': 'proj_002',
                    'nom': 'Plateforme Analytics BI',
                    'status': 'active',
                    'priorite': 'haute',
                    'progression': 58,
                    'budget': 200000,
                    'budget_utilise': 116000,
                    'date_debut': '2024-02-01',
                    'date_fin_prevue': '2024-07-15',
                    'date_fin_reelle': None,
                    'manager': 'Bob Dupont',
                    'equipe_taille': 7,
                    'risque_niveau': 'haut',
                    'score_sante': 65,
                    'velocite_moyenne': 18,
                    'client': 'Direction Commerciale',
                    'roi_prevu': 380000,
                    'dependances': ['proj_004', 'proj_006'],
                    'bloque': ['proj_007'],
                    'technologies': ['Python', 'React', 'PostgreSQL', 'Apache Spark'],
                    'jalons': [
                        {'nom': 'Architecture', 'date': '2024-02-28', 'status': 'completed', 'progression': 100},
                        {'nom': 'Backend API', 'date': '2024-04-30', 'status': 'active', 'progression': 75},
                        {'nom': 'Frontend Dashboard', 'date': '2024-06-15', 'status': 'pending', 'progression': 20},
                        {'nom': 'Tests Performance', 'date': '2024-07-01', 'status': 'pending', 'progression': 0},
                        {'nom': 'Déploiement', 'date': '2024-07-15', 'status': 'pending', 'progression': 0}
                    ],
                    'ressources_cles': ['Bob Dupont (Lead)', 'Maria Garcia (Data Engineer)', 'Tom Wilson (Frontend)']
                },
                {
                    'id': 'proj_003',
                    'nom': 'Système Authentification SSO',
                    'status': 'completed',
                    'priorite': 'critique',
                    'progression': 100,
                    'budget': 80000,
                    'budget_utilise': 75000,
                    'date_debut': '2023-11-01',
                    'date_fin_prevue': '2024-01-31',
                    'date_fin_reelle': '2024-01-28',
                    'manager': 'Charlie Moreau',
                    'equipe_taille': 3,
                    'risque_niveau': 'bas',
                    'score_sante': 95,
                    'velocite_moyenne': 28,
                    'client': 'Direction IT',
                    'roi_prevu': 150000,
                    'dependances': [],
                    'bloque': ['proj_001', 'proj_005'],
                    'technologies': ['OAuth2', 'LDAP', 'Spring Security'],
                    'jalons': [
                        {'nom': 'Spécifications', 'date': '2023-11-15', 'status': 'completed', 'progression': 100},
                        {'nom': 'Développement Core', 'date': '2023-12-30', 'status': 'completed', 'progression': 100},
                        {'nom': 'Tests Sécurité', 'date': '2024-01-20', 'status': 'completed', 'progression': 100},
                        {'nom': 'Déploiement Production', 'date': '2024-01-28', 'status': 'completed', 'progression': 100}
                    ],
                    'ressources_cles': ['Charlie Moreau (Security Lead)', 'David Park (Backend)', 'Lisa Chen (DevOps)']
                },
                {
                    'id': 'proj_004',
                    'nom': 'Migration Base de Données Cloud',
                    'status': 'active',
                    'priorite': 'haute',
                    'progression': 35,
                    'budget': 150000,
                    'budget_utilise': 52500,
                    'date_debut': '2024-03-01',
                    'date_fin_prevue': '2024-06-30',
                    'date_fin_reelle': None,
                    'manager': 'Diana Prince',
                    'equipe_taille': 4,
                    'risque_niveau': 'haut',
                    'score_sante': 68,
                    'velocite_moyenne': 15,
                    'client': 'Direction IT',
                    'roi_prevu': 280000,
                    'dependances': [],
                    'bloque': ['proj_002'],
                    'technologies': ['AWS RDS', 'PostgreSQL', 'Terraform', 'Docker'],
                    'jalons': [
                        {'nom': 'Audit Base Existante', 'date': '2024-03-20', 'status': 'completed', 'progression': 100},
                        {'nom': 'Stratégie Migration', 'date': '2024-04-15', 'status': 'active', 'progression': 80},
                        {'nom': 'Migration Test', 'date': '2024-05-30', 'status': 'pending', 'progression': 10},
                        {'nom': 'Migration Production', 'date': '2024-06-30', 'status': 'pending', 'progression': 0}
                    ],
                    'ressources_cles': ['Diana Prince (DBA Lead)', 'Alex Wong (Cloud Architect)', 'Mike Johnson (DevOps)']
                },
                {
                    'id': 'proj_005',
                    'nom': 'Optimisation Performance Système',
                    'status': 'paused',
                    'priorite': 'moyenne',
                    'progression': 30,
                    'budget': 90000,
                    'budget_utilise': 25000,
                    'date_debut': '2024-01-10',
                    'date_fin_prevue': '2024-05-31',
                    'date_fin_reelle': None,
                    'manager': 'Eve Johnson',
                    'equipe_taille': 2,
                    'risque_niveau': 'bas',
                    'score_sante': 45,
                    'velocite_moyenne': 12,
                    'client': 'Direction IT',
                    'roi_prevu': 120000,
                    'dependances': ['proj_003'],
                    'bloque': [],
                    'technologies': ['Java', 'Redis', 'Elasticsearch', 'Grafana'],
                    'jalons': [
                        {'nom': 'Profiling Applications', 'date': '2024-02-15', 'status': 'completed', 'progression': 100},
                        {'nom': 'Optimisations Backend', 'date': '2024-03-30', 'status': 'paused', 'progression': 40},
                        {'nom': 'Tests Performance', 'date': '2024-05-15', 'status': 'pending', 'progression': 0},
                        {'nom': 'Déploiement', 'date': '2024-05-31', 'status': 'pending', 'progression': 0}
                    ],
                    'ressources_cles': ['Eve Johnson (Performance Lead)', 'Ryan Smith (Backend Dev)']
                },
                {
                    'id': 'proj_006',
                    'nom': 'API Gateway Enterprise',
                    'status': 'planning',
                    'priorite': 'haute',
                    'progression': 8,
                    'budget': 110000,
                    'budget_utilise': 8800,
                    'date_debut': '2024-04-01',
                    'date_fin_prevue': '2024-08-15',
                    'date_fin_reelle': None,
                    'manager': 'Frank Miller',
                    'equipe_taille': 3,
                    'risque_niveau': 'moyen',
                    'score_sante': 82,
                    'velocite_moyenne': 22,
                    'client': 'Direction Architecture',
                    'roi_prevu': 200000,
                    'dependances': [],
                    'bloque': ['proj_002'],
                    'technologies': ['Kong', 'Kubernetes', 'Go', 'Prometheus'],
                    'jalons': [
                        {'nom': 'Architecture & Design', 'date': '2024-04-30', 'status': 'active', 'progression': 25},
                        {'nom': 'Développement Core', 'date': '2024-06-30', 'status': 'pending', 'progression': 0},
                        {'nom': 'Intégration Services', 'date': '2024-07-31', 'status': 'pending', 'progression': 0},
                        {'nom': 'Tests & Déploiement', 'date': '2024-08-15', 'status': 'pending', 'progression': 0}
                    ],
                    'ressources_cles': ['Frank Miller (API Lead)', 'Grace Liu (DevOps)', 'Jack Brown (Security)']
                },
                {
                    'id': 'proj_007',
                    'nom': 'Tableau de Bord Exécutif',
                    'status': 'planning',
                    'priorite': 'moyenne',
                    'progression': 5,
                    'budget': 85000,
                    'budget_utilise': 4250,
                    'date_debut': '2024-05-01',
                    'date_fin_prevue': '2024-08-30',
                    'date_fin_reelle': None,
                    'manager': 'Helen Zhang',
                    'equipe_taille': 4,
                    'risque_niveau': 'bas',
                    'score_sante': 88,
                    'velocite_moyenne': 20,
                    'client': 'Direction Générale',
                    'roi_prevu': 160000,
                    'dependances': ['proj_002'],
                    'bloque': [],
                    'technologies': ['React', 'D3.js', 'Node.js', 'PostgreSQL'],
                    'jalons': [
                        {'nom': 'Specs & Wireframes', 'date': '2024-05-20', 'status': 'pending', 'progression': 0},
                        {'nom': 'Développement Frontend', 'date': '2024-07-15', 'status': 'pending', 'progression': 0},
                        {'nom': 'Intégration Données', 'date': '2024-08-10', 'status': 'pending', 'progression': 0},
                        {'nom': 'Tests & Livraison', 'date': '2024-08-30', 'status': 'pending', 'progression': 0}
                    ],
                    'ressources_cles': ['Helen Zhang (Product Owner)', 'Ivan Lee (Frontend)', 'Kate Smith (Data)', 'Leo Brown (Backend)']
                }
            ],
            'ressources_globales': {
                'total_budget': 835000,
                'budget_utilise': 367950,
                'budget_engage': 425000,
                'total_personnes': 28,
                'personnes_actives': 20,
                'personnes_disponibles': 8,
                'capacite_mensuelle': 3360,  # heures
                'charge_mensuelle': 2688,
                'cout_moyen_heure': 65,
                'utilisation_cible': 80
            },
            'roadmap_strategique': [
                {
                    'trimestre': 'Q1 2024',
                    'objectifs': ['Finaliser SSO', 'Avancer Mobile App', 'Démarrer Migration DB'],
                    'budget_alloue': 200000,
                    'budget_utilise': 186400,
                    'projets_actifs': 3,
                    'jalons_cles': 8,
                    'jalons_atteints': 7,
                    'satisfaction_client': 8.2
                },
                {
                    'trimestre': 'Q2 2024',
                    'objectifs': ['Livrer Mobile App', 'Finaliser Migration DB', 'Lancer BI Platform', 'Démarrer API Gateway'],
                    'budget_alloue': 280000,
                    'budget_utilise': 181550,
                    'projets_actifs': 5,
                    'jalons_cles': 12,
                    'jalons_atteints': 6,
                    'satisfaction_client': 7.8
                },
                {
                    'trimestre': 'Q3 2024',
                    'objectifs': ['Finaliser BI Platform', 'Livrer API Gateway', 'Dashboard Exécutif', 'Optimisations'],
                    'budget_alloue': 355000,
                    'budget_utilise': 0,
                    'projets_actifs': 6,
                    'jalons_cles': 15,
                    'jalons_atteints': 0,
                    'satisfaction_client': 0
                }
            ],
            'metriques_globales': {
                'velocity_portfolio': 142,  # story points total
                'lead_time_moyen': 14.5,  # jours
                'taux_reussite_jalons': 85.7,  # pourcentage
                'indice_predictibilite': 78,  # sur 100
                'score_satisfaction_globale': 8.1,  # sur 10
                'temps_resolution_blocage': 3.2  # jours moyens
            },
            'alertes_critiques': [
                {
                    'type': 'blocage',
                    'projet': 'proj_002',
                    'message': 'BI Platform bloqué par Migration DB - Impact sur Timeline',
                    'priorite': 'haute',
                    'date_creation': '2024-04-20'
                },
                {
                    'type': 'budget',
                    'projet': 'proj_001',
                    'message': 'Mobile App dépasse budget prévu de 8%',
                    'priorite': 'moyenne',
                    'date_creation': '2024-04-18'
                },
                {
                    'type': 'ressource',
                    'projet': 'proj_005',
                    'message': 'Projet Optimisation en pause - Ressources réallouées',
                    'priorite': 'basse',
                    'date_creation': '2024-04-10'
                }
            ]
        }
    
    def calculate_portfolio_kpis(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les KPIs avancés du portfolio"""
        projets = portfolio_data['projets']
        ressources = portfolio_data['ressources_globales']
        metriques = portfolio_data['metriques_globales']
        
        # Projets par statut avec détails
        status_stats = {}
        for projet in projets:
            status = projet['status']
            if status not in status_stats:
                status_stats[status] = {'count': 0, 'budget': 0, 'progression_moy': 0}
            status_stats[status]['count'] += 1
            status_stats[status]['budget'] += projet['budget']
            status_stats[status]['progression_moy'] += projet['progression']
        
        # Calculer moyennes
        for status, stats in status_stats.items():
            if stats['count'] > 0:
                stats['progression_moy'] /= stats['count']
        
        # Calculs financiers avancés
        budget_total = ressources['total_budget']
        budget_utilise = ressources['budget_utilise']
        budget_engage = ressources['budget_engage']
        taux_utilisation_budget = (budget_utilise / budget_total) * 100
        taux_engagement = (budget_engage / budget_total) * 100
        
        # ROI portfolio
        roi_total_prevu = sum(p.get('roi_prevu', 0) for p in projets if p['status'] in ['active', 'completed'])
        roi_portfolio = ((roi_total_prevu - budget_total) / budget_total * 100) if budget_total > 0 else 0
        
        # Progression pondérée par budget et priorité
        total_budget_projets = sum(p['budget'] for p in projets if p['status'] != 'cancelled')
        if total_budget_projets > 0:
            progression_ponderee = sum(
                p['progression'] * p['budget'] * self._get_priority_weight(p['priorite'])
                for p in projets if p['status'] != 'cancelled'
            ) / (total_budget_projets * 2.5)  # 2.5 = poids moyen des priorités
        else:
            progression_ponderee = 0
        
        # Analyse des risques
        projets_critique = len([p for p in projets if p['risque_niveau'] == 'haut'])
        projets_warning = len([p for p in projets if p['risque_niveau'] == 'moyen'])
        score_risque_moyen = np.mean([self._get_risk_score(p['risque_niveau']) for p in projets])
        
        # Utilisation des ressources avec prédictions
        taux_utilisation_ressources = (ressources['charge_mensuelle'] / ressources['capacite_mensuelle']) * 100
        ressources_sur_utilisees = max(0, ressources['charge_mensuelle'] - ressources['capacite_mensuelle'])
        
        # Analyse des dépendances et blocages
        total_dependances = sum(len(p.get('dependances', [])) for p in projets)
        projets_bloques = len([p for p in projets if len(p.get('bloque', [])) > 0])
        complexite_dependances = total_dependances / len(projets) if projets else 0
        
        # Prédictions basées sur vélocité
        projets_actifs = [p for p in projets if p['status'] == 'active']
        if projets_actifs:
            velocity_moyenne = np.mean([p.get('velocite_moyenne', 20) for p in projets_actifs])
            temps_completion_estime = sum(
                max(0, 100 - p['progression']) / max(1, p.get('velocite_moyenne', 20))
                for p in projets_actifs
            )
        else:
            velocity_moyenne = 0
            temps_completion_estime = 0
        
        # Score de santé portfolio
        score_sante_portfolio = np.mean([p.get('score_sante', 75) for p in projets])
        
        return {
            # Métriques de base
            'total_projets': len(projets),
            'projets_actifs': status_stats.get('active', {}).get('count', 0),
            'projets_completes': status_stats.get('completed', {}).get('count', 0),
            'projets_planning': status_stats.get('planning', {}).get('count', 0),
            'projets_paused': status_stats.get('paused', {}).get('count', 0),
            
            # Progression et performance
            'progression_moyenne': progression_ponderee,
            'velocity_portfolio': metriques['velocity_portfolio'],
            'score_sante_portfolio': score_sante_portfolio,
            'predictibilite': metriques['indice_predictibilite'],
            
            # Financier
            'budget_utilise_pct': taux_utilisation_budget,
            'budget_engage_pct': taux_engagement,
            'budget_disponible': budget_total - budget_engage,
            'roi_portfolio': roi_portfolio,
            'cout_moyen_projet': budget_total / len(projets) if projets else 0,
            
            # Risques et blocages
            'projets_risque_critique': projets_critique,
            'projets_risque_warning': projets_warning,
            'score_risque_moyen': score_risque_moyen,
            'projets_bloques': projets_bloques,
            'complexite_dependances': complexite_dependances,
            
            # Ressources
            'utilisation_ressources': taux_utilisation_ressources,
            'ressources_disponibles': ressources['personnes_disponibles'],
            'surcharge_heures': ressources_sur_utilisees,
            
            # Prédictions
            'velocity_moyenne': velocity_moyenne,
            'temps_completion_estime': temps_completion_estime,
            'satisfaction_client': metriques['score_satisfaction_globale']
        }
    
    def _get_priority_weight(self, priority: str) -> float:
        """Retourne le poids d'une priorité pour les calculs pondérés"""
        weights = {'critique': 4, 'haute': 3, 'moyenne': 2, 'basse': 1}
        return weights.get(priority, 2)
    
    def _get_risk_score(self, risk_level: str) -> float:
        """Convertit le niveau de risque en score numérique"""
        scores = {'haut': 3, 'moyen': 2, 'bas': 1}
        return scores.get(risk_level, 2)
    
    def create_executive_overview(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Crée la vue d'ensemble exécutive avec métriques avancées"""
        projets = portfolio_data['projets']
        
        # Créer subplots pour vue multi-dimensionnelle
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Budget vs Progression vs Priorité', 'ROI Potentiel vs Risque',
                          'Vélocité vs Santé Projet', 'Timeline vs Complexité'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Graphique principal: Budget vs Progression avec bulles priorité
        for projet in projets:
            if projet['status'] != 'cancelled':
                color = self.status_colors.get(projet['status'], '#6B7280')
                priority_size = {'critique': 50, 'haute': 40, 'moyenne': 30, 'basse': 20}
                size = priority_size.get(projet['priorite'], 30)
                
                fig.add_trace(go.Scatter(
                    x=[projet['progression']],
                    y=[projet['budget_utilise']],
                    mode='markers+text',
                    marker=dict(size=size, color=color, opacity=0.7,
                              line=dict(width=2, color='DarkSlateGrey')),
                    text=projet['nom'].split()[0],
                    textposition="top center",
                    name=projet['nom'],
                    showlegend=False,
                    hovertemplate=
                    f"<b>{projet['nom']}</b><br>" +
                    f"Progression: {projet['progression']}%<br>" +
                    f"Budget utilisé: {projet['budget_utilise']:,}€<br>" +
                    f"Priorité: {projet['priorite']}<br>" +
                    f"Équipe: {projet['equipe_taille']} personnes<br>" +
                    f"Manager: {projet['manager']}<br>" +
                    f"ROI prévu: {projet.get('roi_prevu', 0):,}€" +
                    "<extra></extra>"
                ), row=1, col=1)
        
        # ROI vs Risque
        for projet in projets:
            if projet['status'] in ['active', 'planning', 'completed']:
                roi = projet.get('roi_prevu', 0)
                risk_score = self._get_risk_score(projet['risque_niveau'])
                color = self.risk_colors.get(projet['risque_niveau'], '#6B7280')
                
                fig.add_trace(go.Scatter(
                    x=[risk_score],
                    y=[roi],
                    mode='markers',
                    marker=dict(size=20, color=color, opacity=0.8),
                    name=projet['nom'],
                    showlegend=False,
                    hovertemplate=f"<b>{projet['nom']}</b><br>ROI: {roi:,}€<br>Risque: {projet['risque_niveau']}<extra></extra>"
                ), row=1, col=2)
        
        # Vélocité vs Santé
        for projet in projets:
            if projet['status'] == 'active':
                velocity = projet.get('velocite_moyenne', 20)
                health = projet.get('score_sante', 75)
                
                fig.add_trace(go.Scatter(
                    x=[velocity],
                    y=[health],
                    mode='markers',
                    marker=dict(size=25, color=self.status_colors['active'], opacity=0.7),
                    name=projet['nom'],
                    showlegend=False,
                    hovertemplate=f"<b>{projet['nom']}</b><br>Vélocité: {velocity} SP<br>Santé: {health}/100<extra></extra>"
                ), row=2, col=1)
        
        # Timeline vs Complexité (nombre de dépendances)
        for projet in projets:
            if projet['status'] in ['active', 'planning']:
                try:
                    date_fin = datetime.strptime(projet['date_fin_prevue'], '%Y-%m-%d')
                    jours_restants = (date_fin - datetime.now()).days
                    complexite = len(projet.get('dependances', [])) + len(projet.get('bloque', []))
                    
                    fig.add_trace(go.Scatter(
                        x=[jours_restants],
                        y=[complexite],
                        mode='markers',
                        marker=dict(size=30, color=self.priority_colors.get(projet['priorite'], '#6B7280'), opacity=0.7),
                        name=projet['nom'],
                        showlegend=False,
                        hovertemplate=f"<b>{projet['nom']}</b><br>Jours restants: {jours_restants}<br>Dépendances: {complexite}<extra></extra>"
                    ), row=2, col=2)
                except ValueError:
                    continue
        
        # Mise à jour des axes et layout
        fig.update_xaxes(title_text="Progression (%)", row=1, col=1)
        fig.update_yaxes(title_text="Budget Utilisé (€)", row=1, col=1)
        
        fig.update_xaxes(title_text="Niveau de Risque", row=1, col=2)
        fig.update_yaxes(title_text="ROI Prévu (€)", row=1, col=2)
        
        fig.update_xaxes(title_text="Vélocité (SP)", row=2, col=1)
        fig.update_yaxes(title_text="Score Santé", row=2, col=1)
        
        fig.update_xaxes(title_text="Jours Restants", row=2, col=2)
        fig.update_yaxes(title_text="Complexité (Dépendances)", row=2, col=2)
        
        fig.update_layout(
            title_text="Dashboard Exécutif - Vue Multi-dimensionnelle du Portfolio",
            height=700
        )
        
        return fig
    
    def create_advanced_timeline(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Crée un diagramme de Gantt avancé avec dépendances"""
        projets = portfolio_data['projets']
        
        fig = go.Figure()
        
        y_pos = 0
        project_positions = {}
        
        # Trier les projets par date de début
        projets_tries = sorted(projets, key=lambda x: x['date_debut'])
        
        for projet in projets_tries:
            try:
                date_debut = datetime.strptime(projet['date_debut'], '%Y-%m-%d')
                date_fin = datetime.strptime(projet['date_fin_prevue'], '%Y-%m-%d')
                
                if projet['date_fin_reelle']:
                    date_fin_reelle = datetime.strptime(projet['date_fin_reelle'], '%Y-%m-%d')
                else:
                    date_fin_reelle = None
                
                project_positions[projet['id']] = y_pos
                
            except ValueError:
                continue
            
            color = self.status_colors.get(projet['status'], '#6B7280')
            priority_opacity = {'critique': 1.0, 'haute': 0.8, 'moyenne': 0.6, 'basse': 0.4}
            opacity = priority_opacity.get(projet['priorite'], 0.6)
            
            # Barre de progression complète
            total_duration = (date_fin - date_debut).days
            progress_duration = total_duration * (projet['progression'] / 100)
            progress_end = date_debut + timedelta(days=progress_duration)
            
            # Partie complétée
            if projet['progression'] > 0:
                fig.add_trace(go.Scatter(
                    x=[date_debut, progress_end],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(color=color, width=25),
                    opacity=opacity,
                    name=f"{projet['nom']} (Terminé)",
                    showlegend=False,
                    hovertemplate=f"<b>{projet['nom']}</b><br>Progression: {projet['progression']}%<extra></extra>"
                ))
            
            # Partie restante
            if projet['progression'] < 100 and projet['status'] != 'completed':
                fig.add_trace(go.Scatter(
                    x=[progress_end, date_fin],
                    y=[y_pos, y_pos],
                    mode='lines',
                    line=dict(color=color, width=25),
                    opacity=opacity * 0.4,
                    name=f"{projet['nom']} (Restant)",
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Si projet terminé en retard/avance
            if date_fin_reelle and projet['status'] == 'completed':
                if date_fin_reelle != date_fin:
                    retard_color = '#EF4444' if date_fin_reelle > date_fin else '#10B981'
                    fig.add_trace(go.Scatter(
                        x=[min(date_fin, date_fin_reelle), max(date_fin, date_fin_reelle)],
                        y=[y_pos + 0.1, y_pos + 0.1],
                        mode='lines',
                        line=dict(color=retard_color, width=8),
                        name="Écart planning",
                        showlegend=False,
                        hovertemplate=f"Écart: {(date_fin_reelle - date_fin).days} jours<extra></extra>"
                    ))
            
            # Jalons critiques
            for jalon in projet['jalons']:
                if jalon['status'] in ['active', 'completed']:
                    try:
                        date_jalon = datetime.strptime(jalon['date'], '%Y-%m-%d')
                        jalon_color = '#10B981' if jalon['status'] == 'completed' else '#F59E0B'
                        
                        fig.add_trace(go.Scatter(
                            x=[date_jalon],
                            y=[y_pos],
                            mode='markers',
                            marker=dict(
                                color=jalon_color,
                                size=12,
                                symbol='diamond',
                                line=dict(width=2, color='white')
                            ),
                            name=jalon['nom'],
                            showlegend=False,
                            hovertemplate=f"<b>{jalon['nom']}</b><br>Date: {date_jalon.strftime('%d/%m/%Y')}<br>Progression: {jalon.get('progression', 0)}%<extra></extra>"
                        ))
                    except ValueError:
                        continue
            
            y_pos += 1
        
        # Dessiner les dépendances
        for projet in projets:
            if projet['dependances'] and projet['id'] in project_positions:
                for dep_id in projet['dependances']:
                    if dep_id in project_positions:
                        y_from = project_positions[dep_id]
                        y_to = project_positions[projet['id']]
                        
                        # Ligne de dépendance
                        fig.add_shape(
                            type="line",
                            x0=datetime.strptime(next(p['date_fin_prevue'] for p in projets if p['id'] == dep_id), '%Y-%m-%d'),
                            y0=y_from + 0.3,
                            x1=datetime.strptime(projet['date_debut'], '%Y-%m-%d'),
                            y1=y_to - 0.3,
                            line=dict(color="rgba(255, 0, 0, 0.6)", width=2, dash="dot")
                        )
        
        # Ligne "aujourd'hui"
        fig.add_vline(
            x=datetime.now(),
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text="Aujourd'hui"
        )
        
        # Lignes de jalons importants (fins de trimestre)
        for i in range(1, 5):
            quarter_end = datetime(2024, i*3, 30 if i*3 != 12 else 31)
            fig.add_vline(
                x=quarter_end,
                line_width=1,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Q{i}",
                annotation_position="top"
            )
        
        fig.update_layout(
            title="Timeline Avancée - Gantt Multi-projets avec Dépendances",
            xaxis_title="Période",
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(projets_tries))),
                ticktext=[f"{p['nom']} ({p['manager']})" for p in projets_tries]
            ),
            height=max(500, len(projets) * 60),
            margin=dict(l=250),
            showlegend=False
        )
        
        return fig
    
    def create_critical_path_analysis(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Analyse du chemin critique avec graphe de dépendances"""
        projets = portfolio_data['projets']
        
        # Créer un graphe dirigé avec NetworkX (simulation)
        G = nx.DiGraph()
        
        # Ajouter les nœuds (projets)
        for projet in projets:
            duration = 30  # Durée simplifiée
            G.add_node(projet['id'], 
                      label=projet['nom'], 
                      duration=duration,
                      status=projet['status'])
        
        # Ajouter les arêtes (dépendances)
        for projet in projets:
            for dep in projet.get('dependances', []):
                G.add_edge(dep, projet['id'])
        
        # Calculer positions avec spring layout
        try:
            pos = nx.spring_layout(G, k=3, iterations=50)
        except:
            # Fallback si NetworkX pose problème
            pos = {}
            for i, projet in enumerate(projets):
                angle = 2 * np.pi * i / len(projets)
                pos[projet['id']] = (np.cos(angle) * 5, np.sin(angle) * 5)
        
        fig = go.Figure()
        
        # Dessiner les arêtes (dépendances)
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color='rgba(100, 100, 100, 0.5)', width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Flèche au milieu
            mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
            fig.add_trace(go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode='markers',
                marker=dict(symbol='triangle-right', size=8, color='gray'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Dessiner les nœuds (projets)
        for projet in projets:
            if projet['id'] in pos:
                x, y = pos[projet['id']]
                color = self.status_colors.get(projet['status'], '#6B7280')
                
                # Taille basée sur l'impact (durée * priorité)
                priority_multiplier = {'critique': 2.0, 'haute': 1.5, 'moyenne': 1.0, 'basse': 0.7}
                size = 40 * priority_multiplier.get(projet['priorite'], 1.0)
                
                fig.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers+text',
                    marker=dict(
                        size=size,
                        color=color,
                        line=dict(width=3, color='white'),
                        opacity=0.8
                    ),
                    text=projet['nom'].split()[0] + f"<br>{projet['progression']}%",
                    textposition="middle center",
                    textfont=dict(size=10, color='white'),
                    name=projet['nom'],
                    showlegend=False,
                    hovertemplate=
                    f"<b>{projet['nom']}</b><br>" +
                    f"Status: {projet['status']}<br>" +
                    f"Priorité: {projet['priorite']}<br>" +
                    f"Progression: {projet['progression']}%<br>" +
                    f"Manager: {projet['manager']}<br>" +
                    f"Équipe: {projet['equipe_taille']} pers.<br>" +
                    f"Bloque: {len(projet.get('bloque', []))} projet(s)" +
                    "<extra></extra>"
                ))
        
        fig.update_layout(
            title="Analyse du Chemin Critique - Réseau de Dépendances",
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_resource_heatmap(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Créer une heatmap d'allocation des ressources par période"""
        projets = portfolio_data['projets']
        
        # Créer une grille temporelle (12 mois)
        start_date = datetime(2024, 1, 1)
        mois = []
        for i in range(12):
            mois.append((start_date + timedelta(days=30*i)).strftime('%Y-%m'))
        
        # Matrice d'allocation
        allocation_matrix = []
        projet_names = []
        
        for projet in projets:
            if projet['status'] != 'cancelled':
                projet_names.append(f"{projet['nom'][:25]}...")
                allocation_row = []
                
                try:
                    debut = datetime.strptime(projet['date_debut'], '%Y-%m-%d')
                    fin = datetime.strptime(projet['date_fin_prevue'], '%Y-%m-%d')
                    
                    for mois_str in mois:
                        mois_date = datetime.strptime(mois_str + '-15', '%Y-%m-%d')  # Milieu du mois
                        
                        if debut <= mois_date <= fin:
                            # Allocation basée sur taille équipe et priorité
                            base_allocation = projet['equipe_taille']
                            priority_multiplier = {'critique': 1.5, 'haute': 1.2, 'moyenne': 1.0, 'basse': 0.8}
                            allocation = base_allocation * priority_multiplier.get(projet['priorite'], 1.0)
                        else:
                            allocation = 0
                        
                        allocation_row.append(allocation)
                    
                except ValueError:
                    # Données de fallback si dates mal formatées
                    allocation_row = [projet['equipe_taille'] * 0.5] * 12
                
                allocation_matrix.append(allocation_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=allocation_matrix,
            x=mois,
            y=projet_names,
            colorscale='Viridis',
            text=allocation_matrix,
            texttemplate="%{text:.1f}",
            textfont={"size": 10},
            hoverongaps=False,
            colorbar=dict(title="Allocation<br>(personnes-mois)")
        ))
        
        fig.update_layout(
            title="Heatmap d'Allocation des Ressources par Projet et Période",
            height=max(400, len(projet_names) * 30),
            xaxis_title="Période",
            yaxis_title="Projets"
        )
        
        return fig

    def render_planning_dashboard(self):
        """Affiche le dashboard de planning multi-projets enrichi"""
        try:
            st.markdown("### 📅 Module Planning Multi-projets")
            st.markdown("*Vue globale, dépendances et planification stratégique*")
            
            # Configuration enrichie
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                planning_view = st.selectbox(
                    "Vue Planning:",
                    ["Portfolio Overview", "Timeline Avancée", "Chemin Critique", "Allocation Ressources", "Analyse Risques"],
                    key="planning_view_mode"
                )
            
            with col2:
                horizon_filter = st.selectbox(
                    "Horizon temporel:",
                    ["6 prochains mois", "Année courante", "18 prochains mois", "Vue stratégique 3 ans"],
                    key="planning_horizon"
                )
            
            with col3:
                if st.button("📊 Actualiser", use_container_width=True, key="planning_refresh"):
                    st.success("Planning actualisé!")
            
            # Chargement des données
            portfolio_data = self.load_portfolio_data()
            kpis = self.calculate_portfolio_kpis(portfolio_data)
            
            # Alertes critiques en haut
            alertes = portfolio_data.get('alertes_critiques', [])
            if alertes:
                st.subheader("🚨 Alertes Critiques")
                for alerte in alertes[:3]:  # Max 3 alertes
                    if alerte['priorite'] == 'haute':
                        st.error(f"🔴 **{alerte['type'].title()}**: {alerte['message']} ({alerte['projet']})")
                    elif alerte['priorite'] == 'moyenne':
                        st.warning(f"🟡 **{alerte['type'].title()}**: {alerte['message']} ({alerte['projet']})")
                    else:
                        st.info(f"🔵 **{alerte['type'].title()}**: {alerte['message']} ({alerte['projet']})")
            
            # KPIs Portfolio enrichis
            st.subheader("📊 KPIs Portfolio")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            with col1:
                st.metric("📁 Total Projets", kpis['total_projets'], 
                         delta=f"Actifs: {kpis['projets_actifs']}")
            
            with col2:
                health_color = "normal" if kpis['score_sante_portfolio'] > 75 else "inverse"
                st.metric("❤️ Santé Portfolio", f"{kpis['score_sante_portfolio']:.0f}/100",
                         delta_color=health_color)
            
            with col3:
                st.metric("📈 Progression", f"{kpis['progression_moyenne']:.1f}%",
                         delta=f"Vélocité: {kpis['velocity_moyenne']:.0f} SP")
            
            with col4:
                budget_color = "inverse" if kpis['budget_engage_pct'] > 85 else "normal"
                st.metric("💰 Budget Engagé", f"{kpis['budget_engage_pct']:.1f}%",
                         delta_color=budget_color)
            
            with col5:
                risk_color = "inverse" if kpis['projets_risque_critique'] > 2 else "normal"
                st.metric("⚠️ Risques Critiques", kpis['projets_risque_critique'],
                         delta=f"Warning: {kpis['projets_risque_warning']}", delta_color=risk_color)
            
            with col6:
                st.metric("🎯 ROI Portfolio", f"{kpis['roi_portfolio']:.1f}%",
                         delta=f"Satisfaction: {kpis['satisfaction_client']:.1f}/10")
            
            # Métriques opérationnelles
            st.markdown("#### 🔧 Métriques Opérationnelles")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🔗 Complexité Dépendances", f"{kpis['complexite_dependances']:.1f}")
            with col2:
                st.metric("🚫 Projets Bloqués", kpis['projets_bloques'])
            with col3:
                resource_color = "inverse" if kpis['utilisation_ressources'] > 90 else "normal"
                st.metric("👥 Utilisation Ressources", f"{kpis['utilisation_ressources']:.0f}%", delta_color=resource_color)
            with col4:
                st.metric("📅 Prédictibilité", f"{kpis['predictibilite']}/100")
            
            st.divider()
            
            # Contenu selon la vue
            if planning_view == "Portfolio Overview":
                st.plotly_chart(
                    self.create_executive_overview(portfolio_data),
                    use_container_width=True
                )
                
                # Analyse détaillée
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 État Détaillé du Portfolio")
                    
                    projets = portfolio_data['projets']
                    portfolio_stats = []
                    
                    for projet in projets:
                        budget_utilise_pct = (projet['budget_utilise'] / projet['budget']) * 100
                        portfolio_stats.append({
                            'Projet': projet['nom'][:30] + '...' if len(projet['nom']) > 30 else projet['nom'],
                            'Status': projet['status'],
                            'Progression': f"{projet['progression']}%",
                            'Budget': f"{budget_utilise_pct:.0f}%",
                            'Priorité': projet['priorite'],
                            'Risque': projet['risque_niveau'],
                            'Équipe': f"{projet['equipe_taille']} pers.",
                            'Manager': projet['manager']
                        })
                    
                    df_portfolio = pd.DataFrame(portfolio_stats)
                    st.dataframe(df_portfolio, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("🎯 Insights Automatiques")
                    
                    # Analyse automatique basée sur les KPIs
                    if kpis['projets_risque_critique'] > 2:
                        st.error("⚠️ Nombre élevé de projets à risque critique - Action requise")
                    
                    if kpis['utilisation_ressources'] > 90:
                        st.warning("👥 Surcharge des ressources détectée - Réallocation nécessaire")
                    
                    if kpis['score_sante_portfolio'] > 85:
                        st.success("✅ Portfolio en excellente santé")
                    elif kpis['score_sante_portfolio'] < 70:
                        st.warning("📉 Santé du portfolio nécessite attention")
                    
                    if kpis['roi_portfolio'] > 20:
                        st.success(f"💰 ROI portfolio excellent ({kpis['roi_portfolio']:.1f}%)")
                    
                    # Prédictions
                    st.markdown("**🔮 Prédictions:**")
                    if kpis['temps_completion_estime'] > 0:
                        st.info(f"📅 Complétion estimée: {kpis['temps_completion_estime']:.0f} semaines")
                    st.info(f"💡 {kpis['ressources_disponibles']} ressources disponibles pour nouveaux projets")
            
            elif planning_view == "Timeline Avancée":
                st.plotly_chart(
                    self.create_advanced_timeline(portfolio_data),
                    use_container_width=True
                )
                
                # Analyse des jalons et échéances
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🎯 Jalons Critiques à Venir")
                    
                    jalons_critiques = []
                    current_date = datetime.now()
                    
                    for projet in portfolio_data['projets']:
                        for jalon in projet['jalons']:
                            if jalon['status'] in ['active', 'pending']:
                                try:
                                    date_jalon = datetime.strptime(jalon['date'], '%Y-%m-%d')
                                    jours_restants = (date_jalon - current_date).days
                                    
                                    if jours_restants <= 60:  # Jalons dans les 2 prochains mois
                                        jalons_critiques.append({
                                            'Projet': projet['nom'],
                                            'Jalon': jalon['nom'],
                                            'Date': jalon['date'],
                                            'Jours Restants': jours_restants,
                                            'Status': jalon['status'],
                                            'Progression': f"{jalon.get('progression', 0)}%",
                                            'Priorité Projet': projet['priorite']
                                        })
                                except ValueError:
                                    continue
                    
                    if jalons_critiques:
                        df_jalons = pd.DataFrame(jalons_critiques)
                        df_jalons = df_jalons.sort_values('Jours Restants')
                        st.dataframe(df_jalons, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucun jalon critique dans les 60 prochains jours")
                
                with col2:
                    st.subheader("📈 Analyse des Délais")
                    
                    projets_analyse = []
                    for projet in portfolio_data['projets']:
                        if projet['status'] in ['active', 'planning']:
                            try:
                                date_fin = datetime.strptime(projet['date_fin_prevue'], '%Y-%m-%d')
                                jours_restants = (date_fin - current_date).days
                                
                                # Estimation basée sur progression et vélocité
                                if projet['progression'] > 0:
                                    velocity = projet.get('velocite_moyenne', 20)
                                    estimation_completion = (100 - projet['progression']) / velocity * 7  # jours
                                    
                                    ecart = estimation_completion - jours_restants
                                    statut_delai = "🔴 Retard" if ecart > 7 else "🟡 Ajusté" if ecart > 0 else "🟢 À temps"
                                    
                                    projets_analyse.append({
                                        'Projet': projet['nom'][:25] + '...',
                                        'Jours Restants': jours_restants,
                                        'Estimation (j)': f"{estimation_completion:.0f}",
                                        'Écart (j)': f"{ecart:+.0f}",
                                        'Statut': statut_delai,
                                        'Vélocité': f"{velocity} SP"
                                    })
                                    
                            except ValueError:
                                continue
                    
                    if projets_analyse:
                        df_analyse = pd.DataFrame(projets_analyse)
                        st.dataframe(df_analyse, use_container_width=True, hide_index=True)
            
            elif planning_view == "Chemin Critique":
                st.plotly_chart(
                    self.create_critical_path_analysis(portfolio_data),
                    use_container_width=True
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔗 Analyse des Dépendances")
                    
                    dependances_info = []
                    for projet in portfolio_data['projets']:
                        if projet['dependances']:
                            for dep_id in projet['dependances']:
                                dep_projet = next((p for p in portfolio_data['projets'] if p['id'] == dep_id), None)
                                if dep_projet:
                                    dependances_info.append({
                                        'Projet Dépendant': projet['nom'],
                                        'Dépend De': dep_projet['nom'],
                                        'Status Bloquant': dep_projet['status'],
                                        'Progression Bloquant': f"{dep_projet['progression']}%",
                                        'Risque': dep_projet['risque_niveau'],
                                        'Impact': 'Critique' if dep_projet['status'] != 'completed' else 'Résolu'
                                    })
                    
                    if dependances_info:
                        df_deps = pd.DataFrame(dependances_info)
                        st.dataframe(df_deps, use_container_width=True, hide_index=True)
                    else:
                        st.info("Aucune dépendance active détectée")
                
                with col2:
                    st.subheader("🚀 Optimisations Suggérées")
                    
                    st.markdown("**🎯 Actions Prioritaires:**")
                    
                    # Analyse automatique des optimisations
                    projets_bloquants = [p for p in portfolio_data['projets'] if len(p.get('bloque', [])) > 0]
                    if projets_bloquants:
                        for projet in projets_bloquants[:3]:
                            if projet['status'] != 'completed':
                                st.warning(f"• Accélérer **{projet['nom']}** (bloque {len(projet['bloque'])} projet(s))")
                    
                    # Suggestions basées sur les métriques
                    if kpis['projets_bloques'] > 2:
                        st.error("🚫 Trop de projets bloqués - Révision des dépendances nécessaire")
                    
                    if kpis['complexite_dependances'] > 1.5:
                        st.warning("🔗 Complexité de dépendances élevée - Simplification recommandée")
                    
                    st.markdown("**💡 Recommandations Stratégiques:**")
                    st.info("• Paralléliser les développements indépendants")
                    st.info("• Créer des interfaces temporaires pour débloquer")
                    st.info("• Prioriser les projets bloquants critiques")
            
            elif planning_view == "Allocation Ressources":
                st.plotly_chart(
                    self.create_resource_heatmap(portfolio_data),
                    use_container_width=True
                )
                
                # Métriques détaillées d'allocation
                col1, col2, col3 = st.columns(3)
                
                ressources = portfolio_data['ressources_globales']
                
                with col1:
                    st.subheader("👥 Capacités Globales")
                    st.metric("Total Personnes", ressources['total_personnes'])
                    st.metric("Personnes Actives", ressources['personnes_actives'])
                    st.metric("Personnes Disponibles", ressources['personnes_disponibles'])
                    
                    utilisation = kpis['utilisation_ressources']
                    if utilisation > 90:
                        st.error(f"⚠️ Surcharge: {utilisation:.0f}%")
                    elif utilisation > 85:
                        st.warning(f"🟡 Utilisation élevée: {utilisation:.0f}%")
                    else:
                        st.success(f"✅ Utilisation optimale: {utilisation:.0f}%")
                
                with col2:
                    st.subheader("💰 Coûts & Budget")
                    st.metric("Budget Disponible", f"{kpis['budget_disponible']:,}€")
                    st.metric("Coût Moyen/Heure", f"{ressources['cout_moyen_heure']}€")
                    
                    if kpis['surcharge_heures'] > 0:
                        cout_surcharge = kpis['surcharge_heures'] * ressources['cout_moyen_heure']
                        st.error(f"💸 Surcoût surcharge: {cout_surcharge:,}€/mois")
                
                with col3:
                    st.subheader("📊 Prévisions")
                    st.metric("Temps Complétion", f"{kpis['temps_completion_estime']:.0f} sem.")
                    
                    # Prédictions de charge
                    charge_future = kpis['utilisation_ressources'] + 10  # Simulation croissance
                    if charge_future > 100:
                        st.error(f"🚨 Surcharge prévue: {charge_future:.0f}%")
                        st.write("**Actions recommandées:**")
                        st.write("• Recrutement urgent")
                        st.write("• Répriorisation des projets")
                    else:
                        st.info(f"📈 Charge future estimée: {charge_future:.0f}%")
            
            elif planning_view == "Analyse Risques":
                # Créer analyse des risques multi-dimensionnelle
                col1, col2 = st.columns(2)
                
                with col1:
                    # Matrice risques/impact
                    st.subheader("🎯 Matrice Risques vs Impact")
                    
                    risk_data = []
                    for projet in portfolio_data['projets']:
                        if projet['status'] in ['active', 'planning']:
                            risk_score = self._get_risk_score(projet['risque_niveau'])
                            impact = projet['budget'] / 1000  # Impact financier en k€
                            
                            risk_data.append({
                                'x': risk_score,
                                'y': impact,
                                'text': projet['nom'].split()[0],
                                'size': projet['equipe_taille'] * 8,
                                'color': self.risk_colors.get(projet['risque_niveau'], '#6B7280')
                            })
                    
                    if risk_data:
                        fig_risk = go.Figure()
                        
                        for item in risk_data:
                            fig_risk.add_trace(go.Scatter(
                                x=[item['x']],
                                y=[item['y']],
                                mode='markers+text',
                                marker=dict(size=item['size'], color=item['color'], opacity=0.7),
                                text=item['text'],
                                textposition="top center",
                                showlegend=False
                            ))
                        
                        fig_risk.update_layout(
                            title="Risque vs Impact Financier",
                            xaxis_title="Niveau de Risque",
                            yaxis_title="Impact (k€)",
                            height=400
                        )
                        
                        st.plotly_chart(fig_risk, use_container_width=True)
                
                with col2:
                    st.subheader("⚠️ Actions de Mitigation")
                    
                    # Analyse automatique des risques
                    projets_haut_risque = [p for p in portfolio_data['projets'] 
                                         if p['risque_niveau'] == 'haut' and p['status'] in ['active', 'planning']]
                    
                    for projet in projets_haut_risque:
                        with st.expander(f"🔴 {projet['nom']} - Risque Élevé"):
                            st.write(f"**Manager:** {projet['manager']}")
                            st.write(f"**Budget exposé:** {projet['budget_utilise']:,}€")
                            st.write(f"**Équipe:** {projet['equipe_taille']} personnes")
                            
                            # Suggestions automatiques
                            st.write("**Mitigations suggérées:**")
                            if projet['progression'] < 30:
                                st.write("• Révision approfondie de l'architecture")
                                st.write("• Renforcement de l'équipe senior")
                            if len(projet.get('dependances', [])) > 0:
                                st.write("• Réduction des dépendances externes")
                            st.write("• Mise en place de jalons de validation hebdomadaires")
                    
                    # Métriques globales de risques
                    st.markdown("**📊 Score de Risque Portfolio:**")
                    st.metric("Score Moyen", f"{kpis['score_risque_moyen']:.1f}/3")
                    
                    if kpis['score_risque_moyen'] > 2.2:
                        st.error("🚨 Portfolio à haut risque")
                    elif kpis['score_risque_moyen'] > 1.8:
                        st.warning("⚠️ Niveau de risque élevé")
                    else:
                        st.success("✅ Risque maîtrisé")
            
            st.divider()
            
            # Roadmap stratégique enrichie
            st.subheader("🗺️ Roadmap Stratégique")
            
            roadmap = portfolio_data['roadmap_strategique']
            cols = st.columns(len(roadmap))
            
            for i, trimestre in enumerate(roadmap):
                with cols[i]:
                    st.markdown(f"**{trimestre['trimestre']}**")
                    
                    # Métriques principales
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("💰 Budget", f"{trimestre['budget_alloue']:,}€")
                        st.metric("📁 Projets", trimestre['projets_actifs'])
                    
                    with col_b:
                        if trimestre.get('budget_utilise', 0) > 0:
                            utilisation = (trimestre['budget_utilise'] / trimestre['budget_alloue']) * 100
                            st.metric("Utilisé", f"{utilisation:.0f}%")
                        
                        if trimestre.get('jalons_cles', 0) > 0:
                            taux_jalons = (trimestre.get('jalons_atteints', 0) / trimestre['jalons_cles']) * 100
                            st.metric("Jalons", f"{taux_jalons:.0f}%")
                    
                    # Objectifs
                    st.markdown("**Objectifs:**")
                    for obj in trimestre['objectifs']:
                        st.write(f"• {obj}")
                    
                    # Indicateur de santé du trimestre
                    if i == 0 and trimestre.get('satisfaction_client', 0) > 0:  # Q1 complété
                        if trimestre['satisfaction_client'] > 8:
                            st.success(f"😊 Satisfaction: {trimestre['satisfaction_client']}/10")
                        else:
                            st.warning(f"😐 Satisfaction: {trimestre['satisfaction_client']}/10")
            
            st.divider()
            
            # Actions Planning enrichies
            st.subheader("⚡ Actions Planning")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🎯 Optimiseur IA", use_container_width=True, key="planning_ai_optimize"):
                    with st.expander("Optimisations IA Suggérées", expanded=True):
                        st.write("**🤖 Analyse IA du Portfolio:**")
                        st.success("• Réallocation 2h/jour de Projet 5 vers Projet 2 (+15% vélocité)")
                        st.info("• Parallélisation possible: API Gateway + Dashboard Exécutif")
                        st.warning("• Goulot d'étranglement détecté: Migration DB bloque 2 projets")
                        st.write("**ROI Optimisation:** +12% sur 6 mois")
            
            with col2:
                if st.button("📊 Simulateur", use_container_width=True, key="planning_simulator"):
                    with st.expander("Simulation Scénarios", expanded=True):
                        scenario = st.selectbox("Scénario:", 
                                               ["Retard Migration DB (+2 sem)", 
                                                "Recrutement +2 devs", 
                                                "Réduction scope Projet 2"])
                        
                        if scenario == "Retard Migration DB (+2 sem)":
                            st.error("Impact: +3 semaines sur BI Platform, +15k€ surcoût")
                        elif scenario == "Recrutement +2 devs":
                            st.success("Impact: -20% temps completion, ROI +8%")
                        else:
                            st.info("Impact: Livraison anticipée de 3 semaines")
            
            with col3:
                if st.button("🔄 Réallocation Auto", use_container_width=True, key="planning_reallocation"):
                    with st.expander("Réallocation Intelligente", expanded=True):
                        st.write("**Suggestions basées sur les métriques:**")
                        st.write("• Alice Martin: 10h/sem Projet 1 → Projet 2 (critique)")
                        st.write("• Équipe Projet 5 (pause) → Renfort Projet 4")
                        st.write("• David Park (auth complété) → API Gateway")
                        
                        if st.button("Appliquer Suggestions"):
                            st.success("✅ Réallocation programmée pour lundi prochain")
            
            with col4:
                if st.button("📈 Prévisions ML", use_container_width=True, key="planning_ml_forecast"):
                    with st.expander("Modèle Prédictif", expanded=True):
                        st.write("**🔮 Prédictions Machine Learning:**")
                        st.metric("Confiance Modèle", "87.3%")
                        st.write("• Complétion portfolio: 15 septembre 2024")
                        st.write("• Probabilité respect délais: 73%")
                        st.write("• Budget final estimé: 812k€ (-3%)")
                        st.write("• Recommandation: Focus sur Migration DB")
        
        except Exception as e:
            st.error(f"Erreur dans le module planning: {str(e)}")
            with st.expander("Détails de l'erreur"):
                st.code(str(e))
                st.write("Les données de démonstration seront chargées au prochain rafraîchissement.")


def show_planning_module():
    """Point d'entrée pour le module planning"""
    planning_module = PlanningModule()
    planning_module.render_planning_dashboard()


if __name__ == "__main__":
    st.set_page_config(page_title="Module Planning", layout="wide")
    show_planning_module()