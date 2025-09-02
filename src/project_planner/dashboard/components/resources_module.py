# Module Ressources pour PlannerIA
# Gestion des équipes, capacités, compétences et planning des ressources

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

class ResourcesModule:
    def __init__(self):
        self.competence_colors = {
            'Développement': '#3B82F6',
            'Design': '#10B981',
            'Management': '#F59E0B', 
            'Marketing': '#EF4444',
            'Analyse': '#8B5CF6',
            'DevOps': '#EC4899',
            'Testing': '#06B6D4',
            'Product': '#F97316',
            'Data': '#84CC16'
        }
        
    def load_resources_data(self, project_id: str = "projet_test") -> Dict[str, Any]:
        """Charge les données des ressources"""
        return {
            'equipes': [
                {
                    'id': 'dev_team',
                    'nom': 'Équipe Développement',
                    'membres': ['Alice Martin', 'Bob Dupont', 'Charlie Moreau', 'David Kim'],
                    'capacite_totale': 160,  # heures/semaine
                    'charge_actuelle': 135,
                    'manager': 'Alice Martin',
                    'competences': ['Développement', 'DevOps', 'Testing'],
                    'projets_assignes': ['PlannerIA', 'Mobile App', 'API Gateway'],
                    'velocite_moyenne': 45,  # story points
                    'satisfaction': 8.2
                },
                {
                    'id': 'design_team', 
                    'nom': 'Équipe Design',
                    'membres': ['Diana Prince', 'Eve Johnson', 'Frank Miller'],
                    'capacite_totale': 120,
                    'charge_actuelle': 95,
                    'manager': 'Diana Prince',
                    'competences': ['Design', 'Marketing', 'Product'],
                    'projets_assignes': ['PlannerIA', 'Brand Refresh'],
                    'velocite_moyenne': 32,
                    'satisfaction': 7.8
                },
                {
                    'id': 'data_team',
                    'nom': 'Équipe Data & Analytics',
                    'membres': ['Grace Chen', 'Hugo Sanchez'],
                    'capacite_totale': 80,
                    'charge_actuelle': 70,
                    'manager': 'Grace Chen',
                    'competences': ['Data', 'Analyse', 'DevOps'],
                    'projets_assignes': ['Analytics Platform', 'PlannerIA'],
                    'velocite_moyenne': 28,
                    'satisfaction': 9.1
                },
                {
                    'id': 'management_team',
                    'nom': 'Management & Product',
                    'membres': ['Ivan Rodriguez', 'Julia Wong'],
                    'capacite_totale': 80,
                    'charge_actuelle': 70,
                    'manager': 'Ivan Rodriguez',
                    'competences': ['Management', 'Product', 'Analyse'],
                    'projets_assignes': ['Portfolio Management'],
                    'velocite_moyenne': 20,
                    'satisfaction': 8.5
                }
            ],
            'membres': [
                {
                    'nom': 'Alice Martin',
                    'role': 'Tech Lead',
                    'equipe': 'dev_team',
                    'niveau_seniority': 'Senior',
                    'capacite': 40,
                    'charge': 38,
                    'competences': ['Développement', 'DevOps', 'Management'],
                    'niveau': {'Développement': 5, 'DevOps': 4, 'Management': 4},
                    'certifications': ['AWS Solutions Architect', 'Scrum Master'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2024-12-23', '2024-12-24', '2024-12-25'],
                    'cout_horaire': 75,
                    'localisation': 'Paris',
                    'remote_ratio': 60,  # % télétravail
                    'satisfaction_projet': 8.5,
                    'derniere_evaluation': '2024-06-15',
                    'objectifs_formation': ['Kubernetes', 'Machine Learning']
                },
                {
                    'nom': 'Bob Dupont',
                    'role': 'Développeur Senior',
                    'equipe': 'dev_team',
                    'niveau_seniority': 'Senior',
                    'capacite': 40,
                    'charge': 37,
                    'competences': ['Développement', 'Testing', 'DevOps'],
                    'niveau': {'Développement': 4, 'Testing': 5, 'DevOps': 3},
                    'certifications': ['Jest Expert', 'Docker Certified'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2024-12-26', '2024-12-27'],
                    'cout_horaire': 68,
                    'localisation': 'Lyon',
                    'remote_ratio': 80,
                    'satisfaction_projet': 7.8,
                    'derniere_evaluation': '2024-05-20',
                    'objectifs_formation': ['Cypress', 'Performance Testing']
                },
                {
                    'nom': 'Charlie Moreau',
                    'role': 'Développeur Junior',
                    'equipe': 'dev_team',
                    'niveau_seniority': 'Junior',
                    'capacite': 40,
                    'charge': 32,
                    'competences': ['Développement', 'Testing'],
                    'niveau': {'Développement': 3, 'Testing': 2},
                    'certifications': ['JavaScript Fundamentals'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': [],
                    'cout_horaire': 45,
                    'localisation': 'Paris',
                    'remote_ratio': 40,
                    'satisfaction_projet': 8.9,
                    'derniere_evaluation': '2024-07-10',
                    'objectifs_formation': ['React Advanced', 'Node.js', 'Git Advanced']
                },
                {
                    'nom': 'David Kim',
                    'role': 'DevOps Engineer',
                    'equipe': 'dev_team',
                    'niveau_seniority': 'Senior',
                    'capacite': 40,
                    'charge': 28,
                    'competences': ['DevOps', 'Testing', 'Développement'],
                    'niveau': {'DevOps': 5, 'Testing': 4, 'Développement': 3},
                    'certifications': ['AWS DevOps Pro', 'Kubernetes Admin', 'Terraform'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2025-01-02', '2025-01-03'],
                    'cout_horaire': 80,
                    'localisation': 'Remote',
                    'remote_ratio': 100,
                    'satisfaction_projet': 8.7,
                    'derniere_evaluation': '2024-04-12',
                    'objectifs_formation': ['Istio', 'Prometheus']
                },
                {
                    'nom': 'Diana Prince',
                    'role': 'UX/UI Lead Designer',
                    'equipe': 'design_team',
                    'niveau_seniority': 'Senior',
                    'capacite': 40,
                    'charge': 35,
                    'competences': ['Design', 'Product', 'Marketing'],
                    'niveau': {'Design': 5, 'Product': 4, 'Marketing': 3},
                    'certifications': ['Google UX Design', 'Adobe Certified Expert'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2024-12-30', '2024-12-31'],
                    'cout_horaire': 72,
                    'localisation': 'Paris',
                    'remote_ratio': 50,
                    'satisfaction_projet': 7.9,
                    'derniere_evaluation': '2024-06-01',
                    'objectifs_formation': ['Design System', 'User Research']
                },
                {
                    'nom': 'Eve Johnson',
                    'role': 'Graphic Designer',
                    'equipe': 'design_team',
                    'niveau_seniority': 'Medior',
                    'capacite': 40,
                    'charge': 32,
                    'competences': ['Design', 'Marketing'],
                    'niveau': {'Design': 4, 'Marketing': 3},
                    'certifications': ['Adobe Creative Suite'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2025-01-06', '2025-01-07'],
                    'cout_horaire': 55,
                    'localisation': 'Bordeaux',
                    'remote_ratio': 70,
                    'satisfaction_projet': 8.2,
                    'derniere_evaluation': '2024-05-15',
                    'objectifs_formation': ['Motion Design', 'Brand Strategy']
                },
                {
                    'nom': 'Frank Miller',
                    'role': 'Product Designer',
                    'equipe': 'design_team',
                    'niveau_seniority': 'Medior',
                    'capacite': 40,
                    'charge': 28,
                    'competences': ['Design', 'Product'],
                    'niveau': {'Design': 4, 'Product': 4},
                    'certifications': ['Figma Expert', 'Design Thinking'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': [],
                    'cout_horaire': 62,
                    'localisation': 'Paris',
                    'remote_ratio': 60,
                    'satisfaction_projet': 8.6,
                    'derniere_evaluation': '2024-07-22',
                    'objectifs_formation': ['Prototyping', 'A/B Testing']
                },
                {
                    'nom': 'Grace Chen',
                    'role': 'Data Scientist Lead',
                    'equipe': 'data_team',
                    'niveau_seniority': 'Senior',
                    'capacite': 40,
                    'charge': 38,
                    'competences': ['Data', 'Analyse', 'Développement'],
                    'niveau': {'Data': 5, 'Analyse': 5, 'Développement': 3},
                    'certifications': ['AWS ML Specialty', 'Google Analytics', 'Tableau Expert'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2025-01-20', '2025-01-21'],
                    'cout_horaire': 78,
                    'localisation': 'Paris',
                    'remote_ratio': 80,
                    'satisfaction_projet': 9.2,
                    'derniere_evaluation': '2024-04-30',
                    'objectifs_formation': ['MLOps', 'Deep Learning']
                },
                {
                    'nom': 'Hugo Sanchez',
                    'role': 'Data Analyst',
                    'equipe': 'data_team',
                    'niveau_seniority': 'Medior',
                    'capacite': 40,
                    'charge': 32,
                    'competences': ['Data', 'Analyse'],
                    'niveau': {'Data': 3, 'Analyse': 4},
                    'certifications': ['SQL Expert', 'Power BI'],
                    'disponibilite': {
                        'Lun': 8, 'Mar': 8, 'Mer': 8, 'Jeu': 8, 'Ven': 8,
                        'Sam': 0, 'Dim': 0
                    },
                    'conges_planifies': ['2025-02-10', '2025-02-11'],
                    'cout_horaire': 58,
                    'localisation': 'Madrid',
                    'remote_ratio': 90,
                    'satisfaction_projet': 8.8,
                    'derniere_evaluation': '2024-06-18',
                    'objectifs_formation': ['Python Advanced', 'Machine Learning']
                }
            ],
            'planning_hebdomadaire': self._generate_weekly_planning(),
            'cout_ressources': {
                'budget_mensuel_ressources': 120000,
                'cout_reel': 89500,
                'cout_previsionnel': 95000,
                'economies_realisees': 5500
            },
            'metriques_equipes': {
                'turnover_rate': 8.5,  # %
                'temps_recrutement_moyen': 45,  # jours
                'satisfaction_moyenne': 8.4,
                'formations_completees': 23,
                'certifications_obtenues': 8
            }
        }
    
    def _generate_weekly_planning(self) -> List[Dict]:
        """Génère un planning hebdomadaire d'exemple"""
        jours = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi']
        membres = [
            'Alice Martin', 'Bob Dupont', 'Charlie Moreau', 'David Kim',
            'Diana Prince', 'Eve Johnson', 'Frank Miller', 'Grace Chen', 'Hugo Sanchez'
        ]
        
        # Charges réalistes par membre
        charges_par_membre = {
            'Alice Martin': [7.5, 8, 7, 8, 7.5],  # Charge management
            'Bob Dupont': [8, 7.5, 8, 7, 8],
            'Charlie Moreau': [6, 7, 8, 7, 6],  # Junior, moins de charge
            'David Kim': [6, 7, 8, 7, 0],  # Congé vendredi
            'Diana Prince': [8, 7, 8, 6, 8],  # Lead, meetings
            'Eve Johnson': [7, 8, 7, 8, 6],
            'Frank Miller': [7, 6, 8, 7, 0],  # Temps partiel vendredi
            'Grace Chen': [8, 8, 7, 8, 7.5],  # Data lead
            'Hugo Sanchez': [7, 8, 7, 6, 6]   # Analyst
        }
        
        # Définir les projets par membre
        projets_par_membre = {
            'Alice Martin': ['PlannerIA Core', 'Team Management', 'Architecture'],
            'Bob Dupont': ['Mobile App', 'Testing Framework', 'Code Review'],
            'Charlie Moreau': ['PlannerIA Frontend', 'Learning', 'Bug Fixes'],
            'David Kim': ['CI/CD Pipeline', 'Infrastructure', 'Monitoring'],
            'Diana Prince': ['UX Research', 'Design System', 'User Testing'],
            'Eve Johnson': ['Brand Assets', 'Marketing Materials', 'Icons'],
            'Frank Miller': ['Product Wireframes', 'Prototyping', 'User Flow'],
            'Grace Chen': ['Analytics Platform', 'ML Models', 'Data Pipeline'],
            'Hugo Sanchez': ['Reporting Dashboards', 'Data Analysis', 'KPIs']
        }
        
        planning = []
        for i, jour in enumerate(jours):
            for membre in membres:
                charge = charges_par_membre[membre][i]
                projets = projets_par_membre.get(membre, ['Projet général'])
                
                planning.append({
                    'jour': jour,
                    'membre': membre,
                    'charge_prevue': 8,
                    'charge_reelle': charge,
                    'disponibilite': 8 - charge,
                    'projets': projets,
                    'efficacite': min(100, (charge / 8) * 120) if charge > 0 else 0,
                    'statut': 'Congé' if charge == 0 else 'Disponible' if charge < 6 else 'Optimal' if charge <= 8 else 'Surchargé'
                })
        
        return planning
    
    def calculate_resources_kpis(self, resources_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calcule les KPIs des ressources"""
        membres = resources_data['membres']
        equipes = resources_data['equipes']
        
        capacite_totale = sum(m['capacite'] for m in membres)
        charge_totale = sum(m['charge'] for m in membres)
        taux_utilisation = (charge_totale / capacite_totale * 100) if capacite_totale > 0 else 0
        
        # Calcul des coûts
        cout_data = resources_data['cout_ressources']
        taux_budget = (cout_data['cout_reel'] / cout_data['budget_mensuel_ressources'] * 100)
        
        # Risques de surcharge (>90% de capacité)
        membres_surcharges = [m for m in membres if (m['charge'] / m['capacite']) > 0.9]
        
        # Disponibilités par compétence
        competences_disponibles = {}
        for membre in membres:
            disponibilite_membre = membre['capacite'] - membre['charge']
            for comp in membre['competences']:
                if comp not in competences_disponibles:
                    competences_disponibles[comp] = 0
                competences_disponibles[comp] += disponibilite_membre
        
        # Satisfaction moyenne
        satisfaction_moyenne = sum(eq['satisfaction'] for eq in equipes) / len(equipes)
        
        # Répartition des niveaux de séniorité
        niveaux_seniority = {}
        for membre in membres:
            niveau = membre.get('niveau_seniority', 'Unknown')
            niveaux_seniority[niveau] = niveaux_seniority.get(niveau, 0) + 1
        
        return {
            'capacite_totale': capacite_totale,
            'charge_totale': charge_totale,
            'taux_utilisation': taux_utilisation,
            'disponibilite': capacite_totale - charge_totale,
            'cout_total': cout_data['cout_reel'],
            'budget_ressources': cout_data['budget_mensuel_ressources'],
            'taux_budget': taux_budget,
            'economies': cout_data.get('economies_realisees', 0),
            'membres_surcharges': len(membres_surcharges),
            'membres_sous_charges': len([m for m in membres if (m['charge'] / m['capacite']) < 0.7]),
            'competences_disponibles': competences_disponibles,
            'satisfaction_moyenne': satisfaction_moyenne,
            'niveaux_seniority': niveaux_seniority,
            'remote_ratio_moyen': sum(m.get('remote_ratio', 50) for m in membres) / len(membres),
            'cout_horaire_moyen': sum(m['cout_horaire'] for m in membres) / len(membres)
        }
    
    def create_team_capacity_chart(self, resources_data: Dict[str, Any]) -> go.Figure:
        """Crée le graphique de capacité des équipes"""
        equipes = resources_data['equipes']
        
        noms = [eq['nom'] for eq in equipes]
        capacites = [eq['capacite_totale'] for eq in equipes]
        charges = [eq['charge_actuelle'] for eq in equipes]
        disponibles = [cap - charge for cap, charge in zip(capacites, charges)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Disponible',
            x=noms,
            y=disponibles,
            marker_color='lightgreen',
            text=[f'+{d}h' for d in disponibles],
            textposition='inside'
        ))
        
        fig.add_trace(go.Bar(
            name='Charge Actuelle',
            x=noms,
            y=charges,
            marker_color='steelblue',
            text=[f'{c}h' for c in charges],
            textposition='inside'
        ))
        
        # Ligne de capacité totale
        fig.add_trace(go.Scatter(
            x=noms,
            y=capacites,
            mode='markers+text',
            name='Capacité Max',
            marker=dict(color='red', size=8, symbol='diamond'),
            text=[f'Max: {c}h' for c in capacites],
            textposition='top center'
        ))
        
        fig.update_layout(
            title="Capacité vs Charge par Équipe (heures/semaine)",
            xaxis_title="Équipes",
            yaxis_title="Heures",
            barmode='stack',
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_skills_matrix(self, resources_data: Dict[str, Any]) -> go.Figure:
        """Crée la matrice des compétences améliorée"""
        membres = resources_data['membres']
        
        # Créer la matrice
        competences_uniques = set()
        for membre in membres:
            competences_uniques.update(membre['competences'])
        
        competences_list = sorted(list(competences_uniques))
        noms_membres = [m['nom'].split()[0] for m in membres]  # Prénom seulement
        
        # Matrice de niveaux avec texte
        matrice = []
        texte_matrice = []
        
        for competence in competences_list:
            row = []
            text_row = []
            for membre in membres:
                niveau = membre.get('niveau', {}).get(competence, 0)
                row.append(niveau)
                if niveau > 0:
                    certifs = [c for c in membre.get('certifications', []) if competence.lower() in c.lower()]
                    text_row.append(f"Niv.{niveau}" + (f"\n{len(certifs)} cert." if certifs else ""))
                else:
                    text_row.append("")
            matrice.append(row)
            texte_matrice.append(text_row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrice,
            x=noms_membres,
            y=competences_list,
            colorscale='RdYlGn',
            text=texte_matrice,
            texttemplate="%{text}",
            textfont={"size": 9},
            hoverongaps=False,
            colorbar=dict(
                title="Niveau",
                tickvals=[0, 1, 2, 3, 4, 5],
                ticktext=["Aucun", "Débutant", "Junior", "Intermédiaire", "Avancé", "Expert"]
            )
        ))
        
        fig.update_layout(
            title="Matrice des Compétences (avec certifications)",
            height=500,
            xaxis_title="Membres de l'équipe",
            yaxis_title="Compétences",
            font=dict(size=11)
        )
        
        return fig
    
    def create_workload_timeline(self, resources_data: Dict[str, Any]) -> go.Figure:
        """Crée la timeline de charge de travail améliorée"""
        planning = resources_data['planning_hebdomadaire']
        df = pd.DataFrame(planning)
        
        # Calcul de moyennes par jour
        charge_moyenne_jour = df.groupby('jour')['charge_reelle'].mean()
        disponibilite_moyenne_jour = df.groupby('jour')['disponibilite'].mean()
        
        jours_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi']
        
        fig = go.Figure()
        
        # Charge moyenne
        fig.add_trace(go.Scatter(
            x=jours_order,
            y=[charge_moyenne_jour[j] for j in jours_order],
            mode='lines+markers',
            name='Charge Moyenne',
            line=dict(color='steelblue', width=4),
            marker=dict(size=10),
            fill='tonexty'
        ))
        
        # Disponibilité moyenne
        fig.add_trace(go.Scatter(
            x=jours_order,
            y=[disponibilite_moyenne_jour[j] for j in jours_order],
            mode='lines+markers',
            name='Disponibilité Moyenne',
            line=dict(color='lightgreen', width=4),
            marker=dict(size=10),
            fill='tozeroy'
        ))
        
        # Ligne de capacité maximale
        fig.add_hline(y=8, line_dash="dash", line_color="red", 
                      annotation_text="Capacité Max (8h)")
        
        fig.update_layout(
            title="Charge vs Disponibilité Moyenne par Jour",
            xaxis_title="Jours de la semaine",
            yaxis_title="Heures",
            height=400,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def create_team_satisfaction_chart(self, resources_data: Dict[str, Any]) -> go.Figure:
        """Crée le graphique de satisfaction des équipes"""
        equipes = resources_data['equipes']
        
        noms = [eq['nom'] for eq in equipes]
        satisfactions = [eq['satisfaction'] for eq in equipes]
        velocites = [eq['velocite_moyenne'] for eq in equipes]
        
        fig = go.Figure()
        
        # Barres de satisfaction
        fig.add_trace(go.Bar(
            name='Satisfaction',
            x=noms,
            y=satisfactions,
            yaxis='y',
            marker_color='lightcoral',
            text=[f'{s}/10' for s in satisfactions],
            textposition='inside'
        ))
        
        # Ligne de vélocité
        fig.add_trace(go.Scatter(
            x=noms,
            y=velocites,
            yaxis='y2',
            mode='lines+markers',
            name='Vélocité (SP)',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Satisfaction d'Équipe vs Vélocité",
            xaxis_title="Équipes",
            yaxis=dict(
                title="Satisfaction (/10)",
                side="left"
            ),
            yaxis2=dict(
                title="Vélocité (Story Points)",
                side="right",
                overlaying="y"
            ),
            height=400,
            legend=dict(x=0.01, y=0.99)
        )
        
        return fig
    
    def create_cost_analysis(self, resources_data: Dict[str, Any]) -> go.Figure:
        """Crée l'analyse des coûts améliorée"""
        membres = resources_data['membres']
        
        # Coût par membre avec détails
        cout_par_membre = []
        for membre in membres:
            cout_mensuel = membre['charge'] * 4.33 * membre['cout_horaire']  # 4.33 semaines/mois
            cout_par_membre.append({
                'membre': membre['nom'].split()[0],  # Prénom seulement
                'cout': cout_mensuel,
                'role': membre['role'],
                'equipe': membre['equipe'],
                'seniority': membre.get('niveau_seniority', 'Unknown'),
                'remote_ratio': membre.get('remote_ratio', 50)
            })
        
        df_cout = pd.DataFrame(cout_par_membre)
        
        # Couleurs par séniorité
        color_map = {'Senior': 'darkblue', 'Medior': 'steelblue', 'Junior': 'lightblue'}
        colors = [color_map.get(row['seniority'], 'gray') for _, row in df_cout.iterrows()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df_cout['membre'],
            y=df_cout['cout'],
            marker_color=colors,
            text=[f"{c:,.0f}€\n{row['seniority']}" for c, (_, row) in zip(df_cout['cout'], df_cout.iterrows())],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Coût: %{y:,.0f}€<br>Remote: %{customdata}%<extra></extra>',
            customdata=df_cout['remote_ratio']
        ))
        
        fig.update_layout(
            title="Coût Mensuel par Ressource (par séniorité)",
            xaxis_title="Membres de l'équipe",
            yaxis_title="Coût mensuel (€)",
            height=400
        )
        
        return fig
    
    def create_capacity_planning_chart(self, resources_data: Dict[str, Any]) -> go.Figure:
        """Crée le graphique de planification des capacités"""
        # Simulation de données futures
        semaines = [f"S{i}" for i in range(1, 13)]  # 12 semaines
        
        kpis = self.calculate_resources_kpis(resources_data)
        capacite_base = kpis['capacite_totale']
        
        # Simulation avec variations saisonnières
        capacites_futures = []
        charges_prevues = []
        
        for i in range(12):
            # Variations saisonnières (congés, etc.)
            variation_conges = -20 if i in [5, 6, 10, 11] else 0  # Congés d'été et d'hiver
            capacite_semaine = capacite_base + variation_conges
            
            # Charge prévue avec croissance projet
            charge_base = kpis['charge_totale']
            croissance = i * 5  # +5h par semaine de croissance
            charge_semaine = min(capacite_semaine, charge_base + croissance)
            
            capacites_futures.append(capacite_semaine)
            charges_prevues.append(charge_semaine)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=semaines,
            y=capacites_futures,
            mode='lines+markers',
            name='Capacité Disponible',
            line=dict(color='green', width=3),
            fill='tonexty'
        ))
        
        fig.add_trace(go.Scatter(
            x=semaines,
            y=charges_prevues,
            mode='lines+markers',
            name='Charge Prévisionnelle',
            line=dict(color='orange', width=3),
            fill='tozeroy'
        ))
        
        # Zone de surcharge
        fig.add_trace(go.Scatter(
            x=semaines,
            y=[max(0, c - cap) for c, cap in zip(charges_prevues, capacites_futures)],
            mode='lines',
            name='Surcharge Prévue',
            line=dict(color='red', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title="Planification des Capacités (12 semaines)",
            xaxis_title="Semaines",
            yaxis_title="Heures",
            height=400,
            hovermode='x unified'
        )
        
        return fig

    def render_resources_dashboard(self, project_id: str = "projet_test"):
        """Affiche le dashboard complet des ressources"""
        st.title("👥 Gestion Ressources")
        st.markdown("*Gestion des équipes, capacités et planning*")
        
        # Configuration
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            view_mode = st.selectbox(
                "Vue Ressources:",
                ["Vue d'ensemble", "Matrice compétences", "Planning hebdo", "Analyse coûts", "Satisfaction équipes", "Planification capacités"],
                key="resources_view_mode"
            )
        
        with col2:
            time_filter = st.selectbox(
                "Période:",
                ["Cette semaine", "Ce mois", "Ce trimestre", "Prévisionnel"],
                key="resources_time_filter"
            )
        
        with col3:
            if st.button("👥 Actualiser", use_container_width=True, key="resources_refresh"):
                st.success("Ressources actualisées!")
        
        # Chargement des données
        resources_data = self.load_resources_data(project_id)
        kpis = self.calculate_resources_kpis(resources_data)
        
        # KPIs Ressources
        st.subheader("📊 KPIs Ressources")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("👥 Capacité Totale", f"{kpis['capacite_totale']}h/sem")
        
        with col2:
            color = "normal" if 75 <= kpis['taux_utilisation'] <= 85 else "inverse"
            st.metric(
                "📈 Taux Utilisation",
                f"{kpis['taux_utilisation']:.1f}%",
                delta="Optimal: 75-85%" if color == "normal" else "Ajustement nécessaire",
                delta_color=color
            )
        
        with col3:
            st.metric(
                "💰 Coût Total",
                f"{kpis['cout_total']:,}€/mois",
                delta=f"Économie: {kpis['economies']:,}€" if kpis['economies'] > 0 else None
            )
        
        with col4:
            st.metric(
                "⚠️ Surcharges",
                kpis['membres_surcharges'],
                delta=f"Sous-charges: {kpis['membres_sous_charges']}"
            )
        
        with col5:
            st.metric(
                "😊 Satisfaction",
                f"{kpis['satisfaction_moyenne']:.1f}/10",
                delta=f"Remote: {kpis['remote_ratio_moyen']:.0f}%"
            )
        
        st.divider()
        
        # Affichage selon la vue sélectionnée
        if view_mode == "Vue d'ensemble":
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    self.create_team_capacity_chart(resources_data),
                    use_container_width=True
                )
            
            with col2:
                st.plotly_chart(
                    self.create_workload_timeline(resources_data),
                    use_container_width=True
                )
            
            # Tableau des équipes détaillé
            st.subheader("📋 État des Équipes")
            equipes_data = []
            for eq in resources_data['equipes']:
                taux = (eq['charge_actuelle'] / eq['capacite_totale'] * 100)
                equipes_data.append({
                    'Équipe': eq['nom'],
                    'Membres': len(eq['membres']),
                    'Capacité': f"{eq['capacite_totale']}h",
                    'Charge': f"{eq['charge_actuelle']}h",
                    'Taux': f"{taux:.1f}%",
                    'Vélocité': f"{eq['velocite_moyenne']} SP",
                    'Satisfaction': f"{eq['satisfaction']}/10",
                    'Projets': len(eq['projets_assignes']),
                    'Status': '🔴 Surchargée' if taux > 90 else '🟡 Attention' if taux > 80 else '✅ Optimale'
                })
            
            st.dataframe(pd.DataFrame(equipes_data), use_container_width=True, hide_index=True)
            
        elif view_mode == "Matrice compétences":
            st.plotly_chart(
                self.create_skills_matrix(resources_data),
                use_container_width=True
            )
            
            # Analyse des compétences avec recommandations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🎯 Analyse des Compétences")
                st.markdown("**🟢 Points Forts:**")
                st.write("• Développement: Excellente couverture (Alice niveau 5, équipe solide)")
                st.write("• Data Science: Expertise de haut niveau (Grace niveau 5)")
                st.write("• Design: Leadership expérimenté (Diana niveau 5)")
                st.write("• DevOps: Spécialiste dédié (David niveau 5)")
                
                st.markdown("**🔴 Gaps Identifiés:**")
                st.write("• Testing: Dépendance critique sur Bob")
                st.write("• Marketing: Niveaux junior à améliorer")
                st.write("• Management: Besoin de backup leaders")
            
            with col2:
                st.subheader("💡 Recommandations")
                st.write("**Formations Prioritaires:**")
                for comp, dispo in kpis['competences_disponibles'].items():
                    if dispo < 20:
                        st.warning(f"• {comp}: Faible disponibilité ({dispo:.0f}h) - Formation recommandée")
                    elif dispo > 50:
                        st.success(f"• {comp}: Bonne disponibilité ({dispo:.0f}h)")
                
                st.write("**Plans de Formation en Cours:**")
                for membre in resources_data['membres']:
                    if membre.get('objectifs_formation'):
                        st.write(f"• **{membre['nom'].split()[0]}**: {', '.join(membre['objectifs_formation'])}")
        
        elif view_mode == "Planning hebdo":
            st.plotly_chart(
                self.create_workload_timeline(resources_data),
                use_container_width=True
            )
            
            # Détail planning avec filtres avancés
            st.subheader("📅 Planning Détaillé")
            planning_df = pd.DataFrame(resources_data['planning_hebdomadaire'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                membre_filter = st.selectbox("Membre:", ['Tous'] + list(planning_df['membre'].unique()))
            with col2:
                statut_filter = st.selectbox("Statut:", ['Tous'] + list(planning_df['statut'].unique()))
            with col3:
                jour_filter = st.selectbox("Jour:", ['Tous'] + list(planning_df['jour'].unique()))
            
            # Appliquer les filtres
            filtered_df = planning_df.copy()
            if membre_filter != 'Tous':
                filtered_df = filtered_df[filtered_df['membre'] == membre_filter]
            if statut_filter != 'Tous':
                filtered_df = filtered_df[filtered_df['statut'] == statut_filter]
            if jour_filter != 'Tous':
                filtered_df = filtered_df[filtered_df['jour'] == jour_filter]
            
            # Afficher avec métriques
            st.dataframe(
                filtered_df[['jour', 'membre', 'charge_prevue', 'charge_reelle', 'disponibilite', 'efficacite', 'statut']],
                use_container_width=True,
                hide_index=True
            )
            
            # Métriques du planning
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Charge Moyenne", f"{filtered_df['charge_reelle'].mean():.1f}h")
            with col2:
                st.metric("Efficacité Moyenne", f"{filtered_df['efficacite'].mean():.1f}%")
            with col3:
                surcharges = len(filtered_df[filtered_df['statut'] == 'Surchargé'])
                st.metric("Jours Surchargés", surcharges)
            with col4:
                disponibles = len(filtered_df[filtered_df['statut'] == 'Disponible'])
                st.metric("Jours Disponibles", disponibles)
        
        elif view_mode == "Analyse coûts":
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    self.create_cost_analysis(resources_data),
                    use_container_width=True
                )
            
            with col2:
                # Analyse des coûts par équipe
                equipes_cout = {}
                for membre in resources_data['membres']:
                    equipe = membre['equipe']
                    cout = membre['charge'] * 4.33 * membre['cout_horaire']
                    if equipe not in equipes_cout:
                        equipes_cout[equipe] = 0
                    equipes_cout[equipe] += cout
                
                # Graphique en secteurs
                fig_pie = px.pie(
                    values=list(equipes_cout.values()),
                    names=list(equipes_cout.keys()),
                    title="Répartition des Coûts par Équipe"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Métriques coûts détaillées
            st.subheader("💰 Analyse Financière Détaillée")
            col1, col2, col3, col4 = st.columns(4)
            
            cout_data = resources_data['cout_ressources']
            with col1:
                st.metric(
                    "💰 Budget Mensuel",
                    f"{cout_data['budget_mensuel_ressources']:,}€"
                )
            
            with col2:
                st.metric(
                    "💸 Coût Réel",
                    f"{cout_data['cout_reel']:,}€",
                    delta=f"{kpis['taux_budget']:.1f}% du budget"
                )
            
            with col3:
                st.metric(
                    "💵 Économies",
                    f"{cout_data['economies_realisees']:,}€",
                    delta="Vs prévisionnel"
                )
            
            with col4:
                st.metric(
                    "💎 Coût/Heure Moyen",
                    f"{kpis['cout_horaire_moyen']:.0f}€",
                    delta=f"Séniorité mixte"
                )
        
        elif view_mode == "Satisfaction équipes":
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    self.create_team_satisfaction_chart(resources_data),
                    use_container_width=True
                )
            
            with col2:
                # Facteurs de satisfaction
                st.subheader("📊 Facteurs de Satisfaction")
                
                # Analyse par télétravail
                remote_satisfaction = []
                for membre in resources_data['membres']:
                    remote_satisfaction.append({
                        'Remote %': membre.get('remote_ratio', 50),
                        'Satisfaction': membre.get('satisfaction_projet', 8.0)
                    })
                
                remote_df = pd.DataFrame(remote_satisfaction)
                
                fig_remote = px.scatter(
                    remote_df, 
                    x='Remote %', 
                    y='Satisfaction',
                    title="Satisfaction vs Télétravail",
                    trendline="ols"
                )
                st.plotly_chart(fig_remote, use_container_width=True)
            
            # Métriques satisfaction
            st.subheader("😊 Métriques Bien-être")
            col1, col2, col3, col4 = st.columns(4)
            
            metriques = resources_data['metriques_equipes']
            with col1:
                st.metric("😊 Satisfaction Moyenne", f"{metriques['satisfaction_moyenne']}/10")
            
            with col2:
                turnover_color = "inverse" if metriques['turnover_rate'] > 15 else "normal"
                st.metric("📉 Turnover Rate", f"{metriques['turnover_rate']}%", delta_color=turnover_color)
            
            with col3:
                st.metric("📚 Formations Complétées", f"{metriques['formations_completees']}")
            
            with col4:
                st.metric("🏆 Certifications", f"{metriques['certifications_obtenues']}")
        
        elif view_mode == "Planification capacités":
            st.plotly_chart(
                self.create_capacity_planning_chart(resources_data),
                use_container_width=True
            )
            
            # Recommandations de planification
            st.subheader("🎯 Recommandations Capacités")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🔮 Prévisions:**")
                st.write("• Période de forte charge prévue: Semaines 8-10")
                st.write("• Risque de surcharge: +15% capacity needed")
                st.write("• Impact congés d'été: -8% capacity S5-S6")
                st.write("• Croissance projet: +5h/semaine")
                
                st.markdown("**⚠️ Alertes:**")
                st.warning("Recrutement recommandé pour S7-S8")
                st.info("Formation cross-skills pour flexibilité")
            
            with col2:
                st.markdown("**💡 Actions Suggérées:**")
                st.write("**Court terme (4 semaines):**")
                st.write("• Redistribuer 10h de Charlie vers Alice")
                st.write("• Former Frank sur les compétences Product")
                st.write("• Planifier les congés Q1 2025")
                
                st.write("**Moyen terme (12 semaines):**")
                st.write("• Recruter 1 Développeur Medior")
                st.write("• Étendre l'équipe Data (1 Analyst)")
                st.write("• Certification DevOps pour Bob")
        
        st.divider()
        
        # Actions rapides améliorées
        st.subheader("⚡ Actions Ressources")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👥 Affecter Ressource", use_container_width=True, key="resources_assign"):
                with st.expander("Affectation Rapide", expanded=True):
                    membre_select = st.selectbox("Membre:", [m['nom'] for m in resources_data['membres']])
                    projet_select = st.selectbox("Projet:", ["PlannerIA", "Mobile App", "Analytics Platform"])
                    heures = st.number_input("Heures/semaine:", min_value=0, max_value=40, value=8)
                    if st.button("Confirmer Affectation"):
                        st.success(f"✅ {membre_select} affecté(e) à {projet_select} ({heures}h/sem)")
        
        with col2:
            if st.button("📊 Plan de Charge", use_container_width=True, key="resources_plan"):
                with st.expander("Planification Charge", expanded=True):
                    st.write("**Optimisations suggérées:**")
                    for membre in resources_data['membres']:
                        taux = (membre['charge'] / membre['capacite']) * 100
                        if taux > 90:
                            st.warning(f"⚠️ {membre['nom']}: Surcharge ({taux:.0f}%)")
                        elif taux < 70:
                            st.info(f"💡 {membre['nom']}: Disponible ({100-taux:.0f}% libre)")
        
        with col3:
            if st.button("🎯 Gérer Compétences", use_container_width=True, key="resources_skills"):
                with st.expander("Gestion Compétences", expanded=True):
                    st.write("**Formations en cours:**")
                    formations_actives = 0
                    for membre in resources_data['membres']:
                        if membre.get('objectifs_formation'):
                            st.write(f"• **{membre['nom']}**: {', '.join(membre['objectifs_formation'][:2])}")
                            formations_actives += len(membre['objectifs_formation'])
                    st.metric("Total formations actives", formations_actives)
        
        with col4:
            if st.button("📅 Planning Congés", use_container_width=True, key="resources_vacation"):
                with st.expander("Calendrier Congés", expanded=True):
                    st.write("**Congés planifiés:**")
                    for membre in resources_data['membres']:
                        if membre.get('conges_planifies'):
                            st.write(f"• **{membre['nom']}**: {', '.join(membre['conges_planifies'][:3])}")
                    
                    # Impact sur capacité
                    total_jours_conges = sum(len(m.get('conges_planifies', [])) for m in resources_data['membres'])
                    impact_capacity = (total_jours_conges * 8) / 7  # Conversion jours -> heures semaine
                    st.metric("Impact capacité semaine", f"-{impact_capacity:.0f}h")


# Fonction d'entrée pour intégration
def show_resources_module(project_id: str = "projet_test"):
    """Point d'entrée pour le module ressources"""
    resources_module = ResourcesModule()
    resources_module.render_resources_dashboard(project_id)


if __name__ == "__main__":
    st.set_page_config(page_title="Module Ressources", layout="wide")
    show_resources_module()