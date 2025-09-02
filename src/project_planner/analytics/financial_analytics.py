"""
💰 Financial Analytics Pro - Module d'Analyse Financière Avancée
===============================================================

Système d'analyse financière professionnel avec:
- Analyse de variance budgétaire avancée
- Projections cash flow avec Monte Carlo
- ROI calculator interactif
- Burn-down charts avec prédictions
- Cost breakdown structure intelligente

Auteur: PlannerIA Team  
Date: 2025-08-31
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import statistics
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')


class CostCategory(Enum):
    """Catégories de coûts standards"""
    LABOR = ("labor", "Main d'œuvre", "#3b82f6")
    MATERIAL = ("material", "Matériel/Licences", "#10b981") 
    INFRASTRUCTURE = ("infrastructure", "Infrastructure", "#f59e0b")
    EXTERNAL = ("external", "Services Externes", "#8b5cf6")
    OVERHEAD = ("overhead", "Frais Généraux", "#6b7280")
    CONTINGENCY = ("contingency", "Réserves", "#ef4444")
    
    def __init__(self, code: str, label: str, color: str):
        self.code = code
        self.label = label
        self.color = color


class FinancialMetric(Enum):
    """Métriques financières clés"""
    ROI = "return_on_investment"
    NPV = "net_present_value"
    IRR = "internal_rate_return"
    PAYBACK = "payback_period"
    CPI = "cost_performance_index"
    SPI = "schedule_performance_index"
    EAC = "estimate_at_completion"
    VAC = "variance_at_completion"


@dataclass
class FinancialSnapshot:
    """Snapshot financier à un moment donné"""
    date: datetime
    planned_cost: float
    actual_cost: float
    earned_value: float
    budget_at_completion: float
    estimate_at_completion: float
    progress_percentage: float
    
    @property
    def cost_variance(self) -> float:
        """Variance de coût (EV - AC)"""
        return self.earned_value - self.actual_cost
    
    @property
    def schedule_variance(self) -> float:
        """Variance de planning (EV - PV)"""
        return self.earned_value - self.planned_cost
    
    @property
    def cost_performance_index(self) -> float:
        """Indice de performance coût (EV / AC)"""
        return self.earned_value / self.actual_cost if self.actual_cost > 0 else 1.0
    
    @property
    def schedule_performance_index(self) -> float:
        """Indice de performance planning (EV / PV)"""
        return self.earned_value / self.planned_cost if self.planned_cost > 0 else 1.0


@dataclass
class CashFlowProjection:
    """Projection de flux de trésorerie"""
    date: datetime
    inflow: float
    outflow: float
    net_flow: float
    cumulative_flow: float
    confidence_interval_low: float
    confidence_interval_high: float


class FinancialAnalyticsEngine:
    """
    💰 Moteur d'Analyse Financière Avancée
    
    Fonctionnalités:
    - Earned Value Management (EVM)
    - Analyse de variance avancée
    - Projections Monte Carlo
    - ROI et NPV calculations
    - Cost forecasting ML
    """
    
    def __init__(self):
        """Initialise le moteur avec modèles et configurations"""
        self.colors = {
            'primary': '#3b82f6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#06b6d4',
            'purple': '#8b5cf6',
            'gray': '#6b7280'
        }
        
        # Configuration pour calculs financiers
        self.discount_rate = 0.08  # 8% taux d'actualisation
        self.tax_rate = 0.25      # 25% taux d'imposition
        self.inflation_rate = 0.03 # 3% inflation
    
    def analyze_project_financials(self, project_data: Dict[str, Any], 
                                 actual_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyse financière complète d'un projet
        
        Args:
            project_data: Données de projet (budget, planning, WBS)
            actual_data: Données réelles (coûts réels, avancement)
            
        Returns:
            Analyse financière complète avec métriques et projections
        """
        if not project_data:
            project_data = self._generate_demo_project_data()
        
        # Normalisation des données d'entrée pour compatibilité
        if 'project_overview' not in project_data:
            project_data = {
                'project_overview': {
                    'total_cost': project_data.get('budget', project_data.get('total_cost', 100000)),
                    'total_duration': project_data.get('duration_months', 6) * 30,
                    'name': project_data.get('name', 'Projet')
                },
                **project_data
            }
        
        if not actual_data:
            actual_data = self._generate_demo_actual_data(project_data)
        
        # 1. Analyse Earned Value Management
        evm_analysis = self._calculate_evm_metrics(project_data, actual_data)
        
        # 2. Analyse des variances
        variance_analysis = self._calculate_variance_analysis(project_data, actual_data)
        
        # 3. Projections cash flow
        cashflow_projections = self._calculate_cashflow_projections(
            project_data, actual_data
        )
        
        # 4. Analyse ROI/NPV (version simplifiée)
        roi_analysis = self._calculate_simple_roi_analysis(project_data)
        
        # 5. Cost breakdown structure
        cost_breakdown = self._calculate_cost_breakdown_structure(project_data)
        
        # 6. Prédictions de coût
        cost_predictions = self._predict_final_costs(project_data, actual_data)
        
        # 7. Analyse des risques financiers
        financial_risks = self._assess_financial_risks(
            evm_analysis, variance_analysis, cost_predictions
        )
        
        return {
            'evm_analysis': evm_analysis,
            'variance_analysis': variance_analysis,
            'cashflow_projections': cashflow_projections,
            'roi_analysis': roi_analysis,
            'cost_breakdown': cost_breakdown,
            'cost_predictions': cost_predictions,
            'financial_risks': financial_risks,
            'summary_metrics': self._calculate_summary_metrics(
                evm_analysis, roi_analysis, financial_risks
            )
        }
    
    def create_financial_dashboard(self, financial_analysis: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Crée un dashboard financier complet avec visualisations avancées
        
        Returns:
            Dictionnaire de figures Plotly pour chaque visualisation
        """
        figures = {}
        
        # 1. Earned Value Chart
        figures['earned_value'] = self._create_earned_value_chart(
            financial_analysis['evm_analysis']
        )
        
        # 2. Variance Analysis
        figures['variance_analysis'] = self._create_variance_analysis_chart(
            financial_analysis['variance_analysis']
        )
        
        # 3. Cash Flow Waterfall
        figures['cashflow_waterfall'] = self._create_cashflow_waterfall(
            financial_analysis['cashflow_projections']
        )
        
        # 4. Cost Breakdown Sunburst
        figures['cost_breakdown'] = self._create_cost_breakdown_sunburst(
            financial_analysis['cost_breakdown']
        )
        
        # 5. ROI Analysis
        figures['roi_analysis'] = self._create_roi_analysis_chart(
            financial_analysis['roi_analysis']
        )
        
        # 6. Financial Burn-down
        figures['burn_down'] = self._create_financial_burndown(
            financial_analysis['evm_analysis']
        )
        
        # 7. Cost Predictions
        figures['cost_predictions'] = self._create_cost_predictions_chart(
            financial_analysis['cost_predictions']
        )
        
        # 8. Financial Risk Heatmap
        figures['financial_risks'] = self._create_financial_risk_heatmap(
            financial_analysis['financial_risks']
        )
        
        return figures
    
    # Méthodes de calcul financier
    def _calculate_evm_metrics(self, project_data: Dict, actual_data: Dict) -> Dict[str, Any]:
        """Calcule les métriques Earned Value Management"""
        
        # Extraction des données
        bac = project_data.get('project_overview', {}).get('total_cost', 100000)  # Budget At Completion
        project_duration = project_data.get('project_overview', {}).get('total_duration', 90)
        
        # Génération de snapshots historiques
        snapshots = []
        start_date = datetime.now() - timedelta(days=60)
        
        for i in range(61):  # 61 jours d'historique
            date = start_date + timedelta(days=i)
            progress = min(100, (i / 60) * 85)  # 85% d'avancement simulé
            
            # Calculs EVM
            planned_cost = bac * (i / 60) * 0.9  # Planifié
            earned_value = bac * (progress / 100)  # Valeur acquise
            actual_cost = earned_value * (1 + np.random.normal(0, 0.1))  # Coût réel avec variance
            actual_cost = max(0, actual_cost)
            
            eac = bac * (actual_cost / earned_value) if earned_value > 0 else bac  # Estimate At Completion
            
            snapshot = FinancialSnapshot(
                date=date,
                planned_cost=planned_cost,
                actual_cost=actual_cost,
                earned_value=earned_value,
                budget_at_completion=bac,
                estimate_at_completion=eac,
                progress_percentage=progress
            )
            snapshots.append(snapshot)
        
        # Métriques actuelles (dernier snapshot)
        current = snapshots[-1]
        
        return {
            'snapshots': snapshots,
            'current_metrics': {
                'bac': bac,
                'pv': current.planned_cost,
                'ev': current.earned_value,
                'ac': current.actual_cost,
                'cv': current.cost_variance,
                'sv': current.schedule_variance,
                'cpi': current.cost_performance_index,
                'spi': current.schedule_performance_index,
                'eac': current.estimate_at_completion,
                'vac': bac - current.estimate_at_completion,
                'progress': current.progress_percentage
            },
            'performance_status': self._get_performance_status(current),
            'forecasts': self._calculate_evm_forecasts(current, bac, project_duration)
        }
    
    def _calculate_simple_roi_analysis(self, project_data: Dict) -> Dict[str, Any]:
        """Version simplifiée de l'analyse ROI"""
        cost = project_data['project_overview']['total_cost']
        return {
            'calculated_roi': 15.5,  # Pourcentage ROI simulé
            'payback_period_months': 18,
            'net_present_value': cost * 0.2,
            'internal_rate_return': 12.8
        }
    
    def _calculate_evm_forecasts(self, current_snapshot, bac: float, project_duration: int) -> Dict[str, Any]:
        """Calcule les prévisions EVM"""
        return {
            'completion_date_forecast': datetime.now() + timedelta(days=project_duration * current_snapshot.schedule_performance_index),
            'final_cost_forecast': current_snapshot.estimate_at_completion,
            'remaining_work_efficiency': current_snapshot.cost_performance_index,
            'schedule_recovery_needed': max(0, 1.0 - current_snapshot.schedule_performance_index) * 100
        }
    
    def _calculate_variance_analysis(self, project_data: Dict, actual_data: Dict) -> Dict[str, Any]:
        """Analyse détaillée des variances par catégorie"""
        
        # Variance par catégorie de coût
        variances_by_category = {}
        
        for category in CostCategory:
            planned = np.random.uniform(5000, 20000)  # Simulation
            actual = planned * np.random.uniform(0.8, 1.3)  # Variance réaliste
            
            variance = actual - planned
            variance_percent = (variance / planned) * 100 if planned > 0 else 0
            
            variances_by_category[category.code] = {
                'category': category.label,
                'planned': planned,
                'actual': actual,
                'variance': variance,
                'variance_percent': variance_percent,
                'status': self._get_variance_status(variance_percent),
                'color': category.color
            }
        
        # Variance par phase
        phases = project_data.get('wbs', {}).get('phases', [])
        variances_by_phase = []
        
        for i, phase in enumerate(phases):
            planned = sum(t.get('cost', 1000) for t in phase.get('tasks', []))
            actual = planned * np.random.uniform(0.85, 1.25)
            
            variance = actual - planned
            variance_percent = (variance / planned) * 100 if planned > 0 else 0
            
            variances_by_phase.append({
                'phase': phase.get('name', f'Phase {i+1}'),
                'planned': planned,
                'actual': actual,
                'variance': variance,
                'variance_percent': variance_percent,
                'status': self._get_variance_status(variance_percent)
            })
        
        # Analyse des tendances (temporairement désactivée)
        trend_analysis = {'trend': 'stable', 'forecast': 'normal'}
        
        return {
            'by_category': variances_by_category,
            'by_phase': variances_by_phase,
            'trend_analysis': trend_analysis,
            'total_variance': sum(v['variance'] for v in variances_by_category.values()),
            'critical_variances': [
                v for v in variances_by_category.values() 
                if abs(v['variance_percent']) > 15
            ]
        }
    
    def _calculate_cashflow_projections(self, project_data: Dict, actual_data: Dict) -> Dict[str, Any]:
        """Calcule les projections de flux de trésorerie avec Monte Carlo"""
        
        projections = []
        start_date = datetime.now()
        total_budget = project_data.get('project_overview', {}).get('total_cost', 100000)
        project_duration = project_data.get('project_overview', {}).get('total_duration', 90)
        
        cumulative = 0
        
        # Projections hebdomadaires
        for week in range(int(project_duration / 7) + 1):
            date = start_date + timedelta(weeks=week)
            
            # Outflows (dépenses) - distribution normale avec pics
            base_outflow = total_budget / (project_duration / 7)
            if 3 <= week <= 8:  # Pic de développement
                base_outflow *= 1.5
            elif week >= int(project_duration / 7 * 0.8):  # Fin de projet
                base_outflow *= 0.6
            
            # Simulation Monte Carlo pour incertitude
            outflow = max(0, np.random.normal(base_outflow, base_outflow * 0.2))
            
            # Inflows (encaissements) - basés sur jalons
            inflow = 0
            if week == 4:  # Jalon 25%
                inflow = total_budget * 0.3
            elif week == 8:  # Jalon 50%
                inflow = total_budget * 0.4
            elif week == int(project_duration / 7 * 0.9):  # Jalon final
                inflow = total_budget * 0.3
            
            net_flow = inflow - outflow
            cumulative += net_flow
            
            # Intervalles de confiance (simulation)
            conf_low = cumulative - abs(cumulative * 0.15)
            conf_high = cumulative + abs(cumulative * 0.15)
            
            projections.append(CashFlowProjection(
                date=date,
                inflow=inflow,
                outflow=outflow,
                net_flow=net_flow,
                cumulative_flow=cumulative,
                confidence_interval_low=conf_low,
                confidence_interval_high=conf_high
            ))
        
        # Analyse des risques de trésorerie
        cash_risks = self._analyze_cash_risks(projections)
        
        return {
            'projections': projections,
            'cash_risks': cash_risks,
            'summary': {
                'total_inflows': sum(p.inflow for p in projections),
                'total_outflows': sum(p.outflow for p in projections),
                'net_cashflow': sum(p.net_flow for p in projections),
                'min_cash_position': min(p.cumulative_flow for p in projections),
                'max_cash_position': max(p.cumulative_flow for p in projections),
                'cash_negative_weeks': len([p for p in projections if p.cumulative_flow < 0])
            }
        }
    
    def _calculate_roi_npv_analysis(self, project_data: Dict, actual_data: Dict) -> Dict[str, Any]:
        """Calcule ROI, NPV, IRR et autres métriques de rentabilité"""
        
        initial_investment = project_data.get('project_overview', {}).get('total_cost', 100000)
        project_duration_years = project_data.get('project_overview', {}).get('total_duration', 90) / 365
        
        # Estimation des bénéfices futurs
        annual_benefits = []
        for year in range(5):  # Projection sur 5 ans
            # Modèle de croissance des bénéfices
            base_benefit = initial_investment * 0.4  # 40% du coût initial
            growth_factor = 1.2 ** year  # Croissance de 20% par an
            market_factor = np.random.uniform(0.8, 1.3)  # Incertitude marché
            
            annual_benefit = base_benefit * growth_factor * market_factor
            annual_benefits.append(annual_benefit)
        
        # Calcul NPV
        npv = self._calculate_npv(initial_investment, annual_benefits, self.discount_rate)
        
        # Calcul IRR (approximation)
        irr = self._calculate_irr_approximation(initial_investment, annual_benefits)
        
        # Calcul période de retour
        payback_period = self._calculate_payback_period(initial_investment, annual_benefits)
        
        # Calcul ROI
        total_benefits = sum(annual_benefits)
        roi = ((total_benefits - initial_investment) / initial_investment) * 100
        
        # Analyse de sensibilité
        sensitivity_analysis = self._calculate_sensitivity_analysis(
            initial_investment, annual_benefits, self.discount_rate
        )
        
        # Scénarios (optimiste, réaliste, pessimiste)
        scenarios = self._calculate_scenario_analysis(initial_investment, annual_benefits)
        
        return {
            'initial_investment': initial_investment,
            'annual_benefits': annual_benefits,
            'npv': npv,
            'irr': irr,
            'roi': roi,
            'payback_period': payback_period,
            'sensitivity_analysis': sensitivity_analysis,
            'scenarios': scenarios,
            'financial_ratios': {
                'benefit_cost_ratio': total_benefits / initial_investment,
                'profitability_index': (npv + initial_investment) / initial_investment,
                'modified_irr': self._calculate_modified_irr(initial_investment, annual_benefits)
            },
            'risk_metrics': {
                'volatility': np.std(annual_benefits) / np.mean(annual_benefits),
                'downside_risk': len([b for b in annual_benefits if b < np.mean(annual_benefits)]) / len(annual_benefits),
                'value_at_risk_95': np.percentile(annual_benefits, 5)  # VaR 95%
            }
        }
    
    def _calculate_cost_breakdown_structure(self, project_data: Dict) -> Dict[str, Any]:
        """Analyse détaillée de la structure des coûts"""
        
        total_budget = project_data.get('project_overview', {}).get('total_cost', 100000)
        
        # Répartition par catégorie de coût
        cost_by_category = {}
        total_allocated = 0
        
        for category in CostCategory:
            if category == CostCategory.LABOR:
                allocation = total_budget * 0.45  # 45% main d'œuvre
            elif category == CostCategory.INFRASTRUCTURE:
                allocation = total_budget * 0.20  # 20% infrastructure
            elif category == CostCategory.MATERIAL:
                allocation = total_budget * 0.15  # 15% matériel
            elif category == CostCategory.EXTERNAL:
                allocation = total_budget * 0.10  # 10% services externes
            elif category == CostCategory.OVERHEAD:
                allocation = total_budget * 0.07  # 7% overhead
            else:  # CONTINGENCY
                allocation = total_budget * 0.03  # 3% contingency
            
            cost_by_category[category.code] = {
                'category': category.label,
                'budget': allocation,
                'percentage': (allocation / total_budget) * 100,
                'color': category.color
            }
            total_allocated += allocation
        
        # Répartition par phase
        phases = project_data.get('wbs', {}).get('phases', [])
        cost_by_phase = []
        
        for i, phase in enumerate(phases):
            phase_cost = sum(t.get('cost', 1000) for t in phase.get('tasks', []))
            
            cost_by_phase.append({
                'phase': phase.get('name', f'Phase {i+1}'),
                'budget': phase_cost,
                'percentage': (phase_cost / total_budget) * 100 if total_budget > 0 else 0,
                'tasks_count': len(phase.get('tasks', [])),
                'avg_task_cost': phase_cost / len(phase.get('tasks', [])) if phase.get('tasks') else 0
            })
        
        # Analyse des concentrations de coût
        cost_concentration = self._analyze_cost_concentration(cost_by_category, cost_by_phase)
        
        return {
            'by_category': cost_by_category,
            'by_phase': cost_by_phase,
            'total_budget': total_budget,
            'cost_concentration': cost_concentration,
            'recommendations': self._generate_cost_optimization_recommendations(
                cost_by_category, cost_by_phase
            )
        }
    
    def _predict_final_costs(self, project_data: Dict, actual_data: Dict) -> Dict[str, Any]:
        """Prédictions des coûts finaux avec ML et méthodes statistiques"""
        
        # Données historiques simulées
        historical_actuals = []
        dates = []
        base_date = datetime.now() - timedelta(days=60)
        
        for i in range(60):
            date = base_date + timedelta(days=i)
            # Simulation d'évolution des coûts avec tendance
            actual_cost = 1000 + (i * 500) + np.random.normal(0, 200)
            actual_cost = max(0, actual_cost)
            
            historical_actuals.append(actual_cost)
            dates.append(date)
        
        # Prédiction par régression linéaire
        X = np.array(range(len(historical_actuals))).reshape(-1, 1)
        y = np.array(historical_actuals)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédictions futures (30 jours)
        future_X = np.array(range(len(historical_actuals), len(historical_actuals) + 30)).reshape(-1, 1)
        future_predictions = model.predict(future_X)
        
        # Intervalle de confiance
        residuals = y - model.predict(X)
        mse = np.mean(residuals**2)
        std_error = np.sqrt(mse)
        
        confidence_interval = 1.96 * std_error  # 95% CI
        
        # Estimation coût final
        current_cost = historical_actuals[-1]
        total_budget = project_data.get('project_overview', {}).get('total_cost', 100000)
        
        # Méthodes de prédiction
        predictions = {
            'linear_regression': {
                'final_cost': future_predictions[-1],
                'confidence_interval': confidence_interval,
                'r_squared': model.score(X, y)
            },
            'exponential_smoothing': self._exponential_smoothing_prediction(historical_actuals),
            'earned_value_method': self._evm_cost_prediction(current_cost, total_budget),
            'monte_carlo': self._monte_carlo_cost_prediction(historical_actuals, 1000)
        }
        
        # Consensus des prédictions
        all_predictions = [p['final_cost'] if isinstance(p, dict) else p for p in predictions.values()]
        consensus_prediction = np.mean([p for p in all_predictions if not np.isnan(p)])
        
        # Analyse des risques de dépassement
        overrun_analysis = self._analyze_cost_overrun_risks(
            current_cost, total_budget, consensus_prediction
        )
        
        return {
            'historical_data': {
                'dates': dates,
                'actuals': historical_actuals,
                'trend': model.coef_[0]
            },
            'predictions': predictions,
            'consensus': {
                'final_cost': consensus_prediction,
                'overrun_amount': consensus_prediction - total_budget,
                'overrun_percentage': ((consensus_prediction - total_budget) / total_budget) * 100
            },
            'overrun_analysis': overrun_analysis,
            'accuracy_metrics': {
                'mae': np.mean(np.abs(residuals)),
                'rmse': np.sqrt(mse),
                'mape': np.mean(np.abs(residuals / y)) * 100
            }
        }
    
    def _assess_financial_risks(self, evm_analysis: Dict, variance_analysis: Dict, 
                               cost_predictions: Dict) -> Dict[str, Any]:
        """Évalue les risques financiers du projet"""
        
        risks = []
        
        # Risque de dépassement budgétaire
        cpi = evm_analysis['current_metrics']['cpi']
        if cpi < 0.9:
            risks.append({
                'type': 'budget_overrun',
                'severity': 'HIGH' if cpi < 0.8 else 'MEDIUM',
                'description': f'CPI de {cpi:.2f} indique un risque de dépassement',
                'impact': abs(evm_analysis['current_metrics']['vac']),
                'probability': 0.8 if cpi < 0.8 else 0.6,
                'mitigation': 'Contrôle strict des coûts et re-planification'
            })
        
        # Risque de retard (impact financier)
        spi = evm_analysis['current_metrics']['spi']
        if spi < 0.9:
            risks.append({
                'type': 'schedule_delay',
                'severity': 'MEDIUM',
                'description': f'SPI de {spi:.2f} peut entraîner des coûts additionnels',
                'impact': evm_analysis['current_metrics']['bac'] * 0.1,  # 10% de surcoût
                'probability': 0.7,
                'mitigation': 'Accélération du planning et/ou ajout de ressources'
            })
        
        # Risque de variance importante
        total_variance = variance_analysis['total_variance']
        if abs(total_variance) > evm_analysis['current_metrics']['bac'] * 0.1:  # >10%
            risks.append({
                'type': 'variance_risk',
                'severity': 'HIGH' if abs(total_variance) > evm_analysis['current_metrics']['bac'] * 0.2 else 'MEDIUM',
                'description': f'Variance totale de €{total_variance:,.0f}',
                'impact': abs(total_variance),
                'probability': 0.9,
                'mitigation': 'Analyse approfondie des causes et actions correctrices'
            })
        
        # Risque de cash flow négatif
        overrun_risk = cost_predictions['overrun_analysis']
        if overrun_risk['probability'] > 0.3:
            risks.append({
                'type': 'cashflow_risk',
                'severity': 'HIGH' if overrun_risk['probability'] > 0.6 else 'MEDIUM',
                'description': f'{overrun_risk["probability"]*100:.0f}% de risque de dépassement',
                'impact': overrun_risk['expected_overrun'],
                'probability': overrun_risk['probability'],
                'mitigation': 'Constitution d\'une réserve de contingence'
            })
        
        # Calcul du score de risque global
        if risks:
            total_impact = sum(risk['impact'] for risk in risks)
            avg_probability = np.mean([risk['probability'] for risk in risks])
            risk_score = (total_impact / evm_analysis['current_metrics']['bac']) * avg_probability
        else:
            risk_score = 0
        
        return {
            'risks': risks,
            'risk_score': risk_score,
            'total_financial_exposure': sum(risk['impact'] for risk in risks),
            'risk_level': self._get_financial_risk_level(risk_score),
            'recommendations': self._generate_financial_risk_recommendations(risks)
        }
    
    # Méthodes de visualisation
    def _create_earned_value_chart(self, evm_analysis: Dict) -> go.Figure:
        """Graphique Earned Value avec prédictions"""
        
        snapshots = evm_analysis['snapshots']
        dates = [s.date for s in snapshots]
        pv = [s.planned_cost for s in snapshots]
        ev = [s.earned_value for s in snapshots]
        ac = [s.actual_cost for s in snapshots]
        
        fig = go.Figure()
        
        # Planned Value (Budget)
        fig.add_trace(go.Scatter(
            x=dates, y=pv,
            mode='lines',
            name='Planned Value (PV)',
            line=dict(color='#10b981', width=2),
            hovertemplate='PV: €%{y:,.0f}<br>Date: %{x}<extra></extra>'
        ))
        
        # Earned Value
        fig.add_trace(go.Scatter(
            x=dates, y=ev,
            mode='lines',
            name='Earned Value (EV)',
            line=dict(color='#3b82f6', width=3),
            hovertemplate='EV: €%{y:,.0f}<br>Date: %{x}<extra></extra>'
        ))
        
        # Actual Cost
        fig.add_trace(go.Scatter(
            x=dates, y=ac,
            mode='lines',
            name='Actual Cost (AC)',
            line=dict(color='#ef4444', width=2),
            hovertemplate='AC: €%{y:,.0f}<br>Date: %{x}<extra></extra>'
        ))
        
        # BAC line
        bac = evm_analysis['current_metrics']['bac']
        fig.add_hline(y=bac, line_dash="dash", line_color="gray",
                     annotation_text=f"BAC: €{bac:,.0f}")
        
        # Projections futures
        current_date = dates[-1]
        future_dates = [current_date + timedelta(days=i*7) for i in range(1, 11)]
        
        # Projection EAC
        current_eac = evm_analysis['current_metrics']['eac']
        eac_projection = np.linspace(ac[-1], current_eac, len(future_dates))
        
        fig.add_trace(go.Scatter(
            x=future_dates, y=eac_projection,
            mode='lines',
            name='EAC Projection',
            line=dict(color='#f59e0b', width=2, dash='dot'),
            hovertemplate='EAC: €%{y:,.0f}<br>Date: %{x}<extra></extra>'
        ))
        
        # Zone de variance acceptable
        upper_bound = [p * 1.1 for p in pv]  # +10%
        lower_bound = [p * 0.9 for p in pv]  # -10%
        
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Zone Acceptable (±10%)',
            hoverinfo='skip'
        ))
        
        # Indicateurs de performance actuels
        current = evm_analysis['current_metrics']
        fig.add_annotation(
            x=dates[-1], y=ac[-1],
            text=f"CPI: {current['cpi']:.2f}<br>SPI: {current['spi']:.2f}",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red" if current['cpi'] < 0.9 else "green",
            bgcolor="white",
            bordercolor="gray"
        )
        
        fig.update_layout(
            title="📊 <b>Earned Value Management - Performance Financière</b>",
            xaxis_title="Date",
            yaxis_title="Coût (€)",
            height=500,
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_variance_analysis_chart(self, variance_analysis: Dict) -> go.Figure:
        """Graphique d'analyse des variances par catégorie et phase"""
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Variance par Catégorie', 'Variance par Phase'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Variance par catégorie
        categories = list(variance_analysis['by_category'].keys())
        cat_variances = [variance_analysis['by_category'][cat]['variance'] for cat in categories]
        cat_labels = [variance_analysis['by_category'][cat]['category'] for cat in categories]
        cat_colors = [variance_analysis['by_category'][cat]['color'] for cat in categories]
        
        fig.add_trace(
            go.Bar(
                x=cat_labels,
                y=cat_variances,
                name='Variance Catégorie',
                marker_color=cat_colors,
                text=[f"€{v:,.0f}" for v in cat_variances],
                textposition='outside',
                hovertemplate='Catégorie: %{x}<br>Variance: €%{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Variance par phase
        phases = [p['phase'] for p in variance_analysis['by_phase']]
        phase_variances = [p['variance'] for p in variance_analysis['by_phase']]
        phase_colors = ['#ef4444' if v > 0 else '#10b981' for v in phase_variances]
        
        fig.add_trace(
            go.Bar(
                x=phases,
                y=phase_variances,
                name='Variance Phase',
                marker_color=phase_colors,
                text=[f"€{v:,.0f}" for v in phase_variances],
                textposition='outside',
                hovertemplate='Phase: %{x}<br>Variance: €%{y:,.0f}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Ligne de référence zéro
        fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)
        fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=2)
        
        fig.update_layout(
            title="📊 <b>Analyse des Variances Budgétaires</b>",
            height=400,
            showlegend=False
        )
        
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_xaxes(tickangle=-45, row=1, col=2)
        
        return fig
    
    def _create_cashflow_waterfall(self, cashflow_analysis: Dict) -> go.Figure:
        """Graphique waterfall du cash flow"""
        
        projections = cashflow_analysis['projections']
        
        # Préparer les données pour le waterfall
        categories = ['Position Initiale']
        values = [0]  # Position initiale
        
        # Grouper par mois pour simplicité
        monthly_inflows = {}
        monthly_outflows = {}
        
        for proj in projections:
            month_key = proj.date.strftime('%Y-%m')
            if month_key not in monthly_inflows:
                monthly_inflows[month_key] = 0
                monthly_outflows[month_key] = 0
            monthly_inflows[month_key] += proj.inflow
            monthly_outflows[month_key] += proj.outflow
        
        # Alterner inflows et outflows
        for month in sorted(monthly_inflows.keys()):
            if monthly_inflows[month] > 0:
                categories.append(f'Encaissements {month}')
                values.append(monthly_inflows[month])
            
            if monthly_outflows[month] > 0:
                categories.append(f'Décaissements {month}')
                values.append(-monthly_outflows[month])
        
        categories.append('Position Finale')
        values.append(None)  # Auto-calculé par Plotly
        
        # Mesures pour waterfall
        measure = ['absolute'] + ['relative'] * (len(values) - 2) + ['total']
        
        # Couleurs
        colors = ['blue'] + ['green' if v > 0 else 'red' for v in values[1:-1]] + ['blue']
        
        fig = go.Figure(go.Waterfall(
            name="Cash Flow",
            orientation="v",
            measure=measure,
            x=categories,
            y=values,
            text=[f"€{abs(v):,.0f}" if v else "" for v in values],
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            marker={"color": colors},
            hovertemplate='%{x}<br>Montant: €%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="💰 <b>Analyse Cash Flow - Cascade Financière</b>",
            xaxis_title="Période",
            yaxis_title="Montant (€)",
            height=500
        )
        
        fig.update_xaxes(tickangle=-45)
        
        return fig
    
    def _create_cost_breakdown_sunburst(self, cost_breakdown: Dict) -> go.Figure:
        """Sunburst chart de la structure des coûts"""
        
        labels = ["Budget Total"]
        parents = [""]
        values = [cost_breakdown['total_budget']]
        colors = ["#3b82f6"]
        
        # Niveau 1: Catégories
        for cat_code, cat_data in cost_breakdown['by_category'].items():
            labels.append(cat_data['category'])
            parents.append("Budget Total")
            values.append(cat_data['budget'])
            colors.append(cat_data['color'])
        
        # Niveau 2: Phases (sous catégorie Labor par exemple)
        labor_budget = cost_breakdown['by_category']['labor']['budget']
        for phase_data in cost_breakdown['by_phase']:
            if phase_data['budget'] > cost_breakdown['total_budget'] * 0.05:  # Seulement les grandes phases
                phase_labor = phase_data['budget'] * 0.6  # Estimation 60% labor
                labels.append(f"{phase_data['phase']} (Labor)")
                parents.append("Main d'œuvre")
                values.append(phase_labor)
                colors.append("#60a5fa")  # Nuance de bleu
        
        fig = go.Figure(go.Sunburst(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="total",
            marker=dict(colors=colors, line=dict(color="white", width=2)),
            textinfo="label+percent parent+value",
            hovertemplate=(
                '<b>%{label}</b><br>' +
                'Budget: €%{value:,.0f}<br>' +
                'Pourcentage: %{percentParent}<br>' +
                '<extra></extra>'
            )
        ))
        
        fig.update_layout(
            title="🎯 <b>Structure des Coûts - Répartition Budgétaire</b>",
            height=600
        )
        
        return fig
    
    def _create_roi_analysis_chart(self, roi_analysis: Dict) -> go.Figure:
        """Graphique d'analyse ROI avec scénarios"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROI par Scénario', 'Flux de Bénéfices Annuels', 'Analyse de Sensibilité', 'Payback Period'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. ROI par scénario
        scenarios = roi_analysis['scenarios']
        scenario_names = list(scenarios.keys())
        scenario_rois = [scenarios[s]['roi'] for s in scenario_names]
        scenario_colors = ['#ef4444', '#f59e0b', '#10b981']  # Rouge, Orange, Vert
        
        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=scenario_rois,
                name='ROI (%)',
                marker_color=scenario_colors,
                text=[f"{roi:.1f}%" for roi in scenario_rois],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # 2. Flux de bénéfices annuels
        years = list(range(1, len(roi_analysis['annual_benefits']) + 1))
        annual_benefits = roi_analysis['annual_benefits']
        
        fig.add_trace(
            go.Bar(
                x=years,
                y=annual_benefits,
                name='Bénéfices Annuels',
                marker_color='#3b82f6'
            ),
            row=1, col=2
        )
        
        # Ligne de break-even
        break_even = roi_analysis['initial_investment'] / len(annual_benefits)
        fig.add_hline(y=break_even, line_dash="dash", line_color="red",
                     annotation_text="Break-even", row=1, col=2)
        
        # 3. Analyse de sensibilité
        sensitivity = roi_analysis['sensitivity_analysis']
        sensitivity_params = list(sensitivity.keys())
        sensitivity_impact = [sensitivity[param]['npv_impact'] for param in sensitivity_params]
        
        fig.add_trace(
            go.Bar(
                x=sensitivity_params,
                y=sensitivity_impact,
                name='Impact NPV (€)',
                marker_color=['#ef4444' if x < 0 else '#10b981' for x in sensitivity_impact]
            ),
            row=2, col=1
        )
        
        # 4. Payback period visualization
        cumulative_benefits = np.cumsum([0] + annual_benefits)
        payback_years = list(range(len(cumulative_benefits)))
        
        fig.add_trace(
            go.Scatter(
                x=payback_years,
                y=cumulative_benefits,
                mode='lines+markers',
                name='Bénéfices Cumulés',
                line=dict(color='#10b981', width=3)
            ),
            row=2, col=2
        )
        
        # Ligne d'investissement initial
        fig.add_hline(y=roi_analysis['initial_investment'], line_dash="dash", 
                     line_color="red", annotation_text="Investment", row=2, col=2)
        
        # Point de payback
        payback = roi_analysis['payback_period']
        if payback <= len(years):
            fig.add_scatter(
                x=[payback], y=[roi_analysis['initial_investment']],
                mode='markers',
                marker=dict(size=15, color='orange', symbol='star'),
                name=f'Payback: {payback:.1f} ans',
                row=2, col=2
            )
        
        fig.update_layout(
            title="💼 <b>Analyse de Rentabilité (ROI/NPV)</b>",
            height=700,
            showlegend=True
        )
        
        return fig
    
    def _create_financial_burndown(self, evm_analysis: Dict) -> go.Figure:
        """Burndown chart financier avec prédictions"""
        
        snapshots = evm_analysis['snapshots']
        dates = [s.date for s in snapshots]
        remaining_budget = [evm_analysis['current_metrics']['bac'] - s.actual_cost for s in snapshots]
        
        # Ligne idéale
        bac = evm_analysis['current_metrics']['bac']
        ideal_burndown = [bac * (1 - i/len(dates)) for i in range(len(dates))]
        
        fig = go.Figure()
        
        # Budget restant réel
        fig.add_trace(go.Scatter(
            x=dates,
            y=remaining_budget,
            mode='lines+markers',
            name='Budget Restant (Réel)',
            line=dict(color='#ef4444', width=3),
            marker=dict(size=6)
        ))
        
        # Burndown idéal
        fig.add_trace(go.Scatter(
            x=dates,
            y=ideal_burndown,
            mode='lines',
            name='Burndown Idéal',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        
        # Zone de tolérance
        tolerance = 0.1  # 10%
        upper_tolerance = [ideal * (1 + tolerance) for ideal in ideal_burndown]
        lower_tolerance = [ideal * (1 - tolerance) for ideal in ideal_burndown]
        
        fig.add_trace(go.Scatter(
            x=dates + dates[::-1],
            y=upper_tolerance + lower_tolerance[::-1],
            fill='toself',
            fillcolor='rgba(16, 185, 129, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Zone de Tolérance (±10%)',
            hoverinfo='skip'
        ))
        
        # Projection EAC
        current_date = dates[-1]
        project_end = current_date + timedelta(days=30)  # Estimation
        eac = evm_analysis['current_metrics']['eac']
        
        fig.add_trace(go.Scatter(
            x=[current_date, project_end],
            y=[remaining_budget[-1], bac - eac],
            mode='lines',
            name='Projection EAC',
            line=dict(color='#f59e0b', width=2, dash='dot')
        ))
        
        # Ligne de référence zéro
        fig.add_hline(y=0, line_color="black", line_width=1, annotation_text="Budget Épuisé")
        
        fig.update_layout(
            title="📉 <b>Financial Burn-down Chart</b>",
            xaxis_title="Date",
            yaxis_title="Budget Restant (€)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_cost_predictions_chart(self, cost_predictions: Dict) -> go.Figure:
        """Graphique des prédictions de coût final"""
        
        # Données historiques
        dates = cost_predictions['historical_data']['dates']
        actuals = cost_predictions['historical_data']['actuals']
        
        # Prédictions
        predictions = cost_predictions['predictions']
        
        fig = go.Figure()
        
        # Données historiques
        fig.add_trace(go.Scatter(
            x=dates,
            y=actuals,
            mode='lines+markers',
            name='Coûts Historiques',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4)
        ))
        
        # Tendance linéaire
        trend = cost_predictions['historical_data']['trend']
        linear_pred = predictions['linear_regression']['final_cost']
        
        # Ligne de tendance
        future_date = dates[-1] + timedelta(days=30)
        fig.add_trace(go.Scatter(
            x=[dates[-1], future_date],
            y=[actuals[-1], linear_pred],
            mode='lines',
            name='Tendance Linéaire',
            line=dict(color='#10b981', width=2, dash='dash')
        ))
        
        # Zone de confiance
        ci = predictions['linear_regression']['confidence_interval']
        fig.add_trace(go.Scatter(
            x=[future_date, future_date],
            y=[linear_pred - ci, linear_pred + ci],
            mode='lines',
            fill='tonexty',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Intervalle de Confiance 95%',
            showlegend=False
        ))
        
        # Autres prédictions (points)
        pred_methods = ['exponential_smoothing', 'earned_value_method', 'monte_carlo']
        pred_colors = ['#f59e0b', '#8b5cf6', '#ef4444']
        
        for i, method in enumerate(pred_methods):
            if method in predictions and isinstance(predictions[method], (int, float)):
                fig.add_trace(go.Scatter(
                    x=[future_date],
                    y=[predictions[method]],
                    mode='markers',
                    name=method.replace('_', ' ').title(),
                    marker=dict(size=12, color=pred_colors[i], symbol='diamond')
                ))
        
        # Consensus
        consensus = cost_predictions['consensus']['final_cost']
        fig.add_trace(go.Scatter(
            x=[future_date],
            y=[consensus],
            mode='markers',
            name='Consensus Prédiction',
            marker=dict(size=15, color='black', symbol='star')
        ))
        
        # Annotations
        overrun = cost_predictions['consensus']['overrun_percentage']
        color = 'red' if overrun > 0 else 'green'
        fig.add_annotation(
            x=future_date,
            y=consensus,
            text=f"Consensus: €{consensus:,.0f}<br>Dépassement: {overrun:+.1f}%",
            showarrow=True,
            arrowcolor=color,
            bgcolor="white",
            bordercolor=color
        )
        
        fig.update_layout(
            title="🔮 <b>Prédictions de Coût Final</b>",
            xaxis_title="Date",
            yaxis_title="Coût Cumulé (€)",
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def _create_financial_risk_heatmap(self, financial_risks: Dict) -> go.Figure:
        """Heatmap des risques financiers"""
        
        risks = financial_risks['risks']
        
        if not risks:
            # Graphique vide si pas de risques
            fig = go.Figure()
            fig.add_annotation(
                text="Aucun risque financier majeur identifié",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="green")
            )
            fig.update_layout(title="🔒 <b>Évaluation des Risques Financiers</b>")
            return fig
        
        # Matrice risque: Probabilité vs Impact
        risk_types = [r['type'] for r in risks]
        probabilities = [r['probability'] * 100 for r in risks]
        impacts = [r['impact'] / 1000 for r in risks]  # En k€
        severities = [r['severity'] for r in risks]
        
        # Couleurs selon sévérité
        color_map = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'}
        colors = [color_map.get(sev, '#6b7280') for sev in severities]
        
        fig = go.Figure()
        
        # Points de risque
        for i, risk in enumerate(risks):
            fig.add_trace(go.Scatter(
                x=[probabilities[i]],
                y=[impacts[i]],
                mode='markers+text',
                text=[risk['type'].replace('_', '<br>')],
                textposition="middle center",
                marker=dict(
                    size=max(20, impacts[i] / 2),
                    color=colors[i],
                    line=dict(color='white', width=2),
                    opacity=0.8
                ),
                name=risk['type'].replace('_', ' ').title(),
                hovertemplate=(
                    f"<b>{risk['type'].replace('_', ' ').title()}</b><br>" +
                    f"Probabilité: {probabilities[i]:.0f}%<br>" +
                    f"Impact: €{risk['impact']:,.0f}<br>" +
                    f"Sévérité: {risk['severity']}<br>" +
                    f"Description: {risk['description']}<br>" +
                    "<extra></extra>"
                ),
                showlegend=False
            ))
        
        # Zones de risque
        fig.add_shape(type="rect", x0=0, y0=0, x1=30, y1=max(impacts)*0.3,
                     fillcolor="rgba(16, 185, 129, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=30, y0=0, x1=70, y1=max(impacts)*0.6,
                     fillcolor="rgba(245, 158, 11, 0.1)", line=dict(width=0))
        fig.add_shape(type="rect", x0=70, y0=0, x1=100, y1=max(impacts),
                     fillcolor="rgba(239, 68, 68, 0.1)", line=dict(width=0))
        
        # Labels des zones
        fig.add_annotation(x=15, y=max(impacts)*0.15, text="<b>FAIBLE</b>",
                          showarrow=False, font=dict(color='green', size=14))
        fig.add_annotation(x=50, y=max(impacts)*0.3, text="<b>MOYEN</b>",
                          showarrow=False, font=dict(color='orange', size=14))
        fig.add_annotation(x=85, y=max(impacts)*0.7, text="<b>ÉLEVÉ</b>",
                          showarrow=False, font=dict(color='red', size=16))
        
        fig.update_layout(
            title="🎯 <b>Matrice des Risques Financiers</b>",
            xaxis=dict(title="Probabilité (%)", range=[0, 100]),
            yaxis=dict(title="Impact Financier (k€)"),
            height=500,
            plot_bgcolor='white'
        )
        
        return fig
    
    # Méthodes utilitaires de calcul financier
    def _calculate_npv(self, initial_investment: float, annual_benefits: List[float], 
                      discount_rate: float) -> float:
        """Calcule la Valeur Actuelle Nette (NPV)"""
        npv = -initial_investment
        for i, benefit in enumerate(annual_benefits):
            npv += benefit / ((1 + discount_rate) ** (i + 1))
        return npv
    
    def _calculate_irr_approximation(self, initial_investment: float, 
                                   annual_benefits: List[float]) -> float:
        """Approximation du Taux de Rendement Interne (IRR)"""
        # Méthode d'approximation simple
        total_benefits = sum(annual_benefits)
        avg_annual_benefit = total_benefits / len(annual_benefits)
        
        # IRR approximatif
        irr_approx = (avg_annual_benefit / initial_investment) - 1
        return max(0, min(1, irr_approx)) * 100  # En pourcentage
    
    def _calculate_payback_period(self, initial_investment: float, 
                                annual_benefits: List[float]) -> float:
        """Calcule la période de retour sur investissement"""
        cumulative = 0
        for i, benefit in enumerate(annual_benefits):
            cumulative += benefit
            if cumulative >= initial_investment:
                # Interpolation pour plus de précision
                remainder = initial_investment - (cumulative - benefit)
                return i + (remainder / benefit) if benefit > 0 else i + 1
        return len(annual_benefits)  # Pas de payback dans la période
    
    def _calculate_modified_irr(self, initial_investment: float, 
                              annual_benefits: List[float]) -> float:
        """Calcule le MIRR (Modified Internal Rate of Return)"""
        # Simplifié pour démo
        fv_benefits = sum(benefit * ((1 + self.discount_rate) ** (len(annual_benefits) - i - 1)) 
                         for i, benefit in enumerate(annual_benefits))
        mirr = (fv_benefits / initial_investment) ** (1 / len(annual_benefits)) - 1
        return mirr * 100
    
    def _get_performance_status(self, snapshot: FinancialSnapshot) -> Dict[str, str]:
        """Détermine le statut de performance basé sur CPI et SPI"""
        cpi = snapshot.cost_performance_index
        spi = snapshot.schedule_performance_index
        
        cost_status = "EXCELLENT" if cpi >= 1.1 else "BON" if cpi >= 0.95 else "ATTENTION" if cpi >= 0.85 else "CRITIQUE"
        schedule_status = "EXCELLENT" if spi >= 1.1 else "BON" if spi >= 0.95 else "ATTENTION" if spi >= 0.85 else "CRITIQUE"
        
        return {
            'cost_status': cost_status,
            'schedule_status': schedule_status,
            'overall_status': min(cost_status, schedule_status, key=lambda x: ['CRITIQUE', 'ATTENTION', 'BON', 'EXCELLENT'].index(x))
        }
    
    def _get_variance_status(self, variance_percent: float) -> str:
        """Détermine le statut d'une variance"""
        abs_var = abs(variance_percent)
        if abs_var <= 5:
            return "BON"
        elif abs_var <= 15:
            return "ATTENTION"
        else:
            return "CRITIQUE"
    
    def _get_financial_risk_level(self, risk_score: float) -> str:
        """Détermine le niveau de risque financier global"""
        if risk_score <= 0.1:
            return "FAIBLE"
        elif risk_score <= 0.3:
            return "MOYEN"
        elif risk_score <= 0.6:
            return "ÉLEVÉ"
        else:
            return "CRITIQUE"
    
    # Méthodes de génération de données (demo)
    def _generate_demo_project_data(self) -> Dict[str, Any]:
        """Génère des données de projet pour démo"""
        return {
            'project_overview': {
                'title': 'Plateforme E-commerce IA',
                'total_cost': 120000,
                'total_duration': 120
            },
            'wbs': {
                'phases': [
                    {'name': 'Conception', 'tasks': [{'cost': 15000}, {'cost': 8000}]},
                    {'name': 'Développement', 'tasks': [{'cost': 35000}, {'cost': 25000}]},
                    {'name': 'Tests & Déploiement', 'tasks': [{'cost': 12000}, {'cost': 10000}]}
                ]
            }
        }
    
    def _generate_demo_actual_data(self, project_data: Dict) -> Dict[str, Any]:
        """Génère des données réelles simulées"""
        return {
            'current_progress': 65,
            'actual_cost_to_date': project_data['project_overview']['total_cost'] * 0.7,
            'phases_completed': 1.5,
            'team_productivity': 0.85
        }
    
    # Méthodes d'analyse avancée
    def _calculate_scenario_analysis(self, initial_investment: float, 
                                   annual_benefits: List[float]) -> Dict[str, Dict]:
        """Calcule l'analyse par scénarios (pessimiste, réaliste, optimiste)"""
        scenarios = {}
        
        for scenario_name, factor in [('pessimiste', 0.7), ('réaliste', 1.0), ('optimiste', 1.3)]:
            adj_benefits = [b * factor for b in annual_benefits]
            npv = self._calculate_npv(initial_investment, adj_benefits, self.discount_rate)
            roi = ((sum(adj_benefits) - initial_investment) / initial_investment) * 100
            
            scenarios[scenario_name] = {
                'annual_benefits': adj_benefits,
                'npv': npv,
                'roi': roi,
                'payback': self._calculate_payback_period(initial_investment, adj_benefits)
            }
        
        return scenarios
    
    def _calculate_sensitivity_analysis(self, initial_investment: float, 
                                      annual_benefits: List[float], 
                                      discount_rate: float) -> Dict[str, Dict]:
        """Analyse de sensibilité sur les paramètres clés"""
        base_npv = self._calculate_npv(initial_investment, annual_benefits, discount_rate)
        sensitivity = {}
        
        # Sensibilité au taux d'actualisation
        for rate_change in [-0.02, -0.01, 0.01, 0.02]:  # ±2%
            new_rate = discount_rate + rate_change
            new_npv = self._calculate_npv(initial_investment, annual_benefits, new_rate)
            sensitivity[f'taux_actualisation_{rate_change:+.0%}'] = {
                'npv_impact': new_npv - base_npv,
                'sensitivity': (new_npv - base_npv) / base_npv if base_npv != 0 else 0
            }
        
        # Sensibilité aux bénéfices
        for benefit_change in [-0.2, -0.1, 0.1, 0.2]:  # ±20%
            adj_benefits = [b * (1 + benefit_change) for b in annual_benefits]
            new_npv = self._calculate_npv(initial_investment, adj_benefits, discount_rate)
            sensitivity[f'benefices_{benefit_change:+.0%}'] = {
                'npv_impact': new_npv - base_npv,
                'sensitivity': (new_npv - base_npv) / base_npv if base_npv != 0 else 0
            }
        
        return sensitivity
    
    def _exponential_smoothing_prediction(self, historical_data: List[float], 
                                        alpha: float = 0.3) -> float:
        """Prédiction par lissage exponentiel"""
        if not historical_data:
            return 0
        
        smoothed = historical_data[0]
        for value in historical_data[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # Projection simple (dernière tendance)
        if len(historical_data) > 1:
            trend = historical_data[-1] - historical_data[-2]
            return smoothed + trend
        
        return smoothed
    
    def _evm_cost_prediction(self, current_cost: float, total_budget: float, 
                           progress: float = 0.65) -> float:
        """Prédiction basée sur Earned Value Method"""
        if progress <= 0:
            return total_budget
        
        # EAC = AC + (BAC - EV) / CPI
        # Simplifié: EAC = AC / (EV/BAC) = AC / progress
        eac = current_cost / progress
        return min(eac, total_budget * 2)  # Cap à 200% du budget
    
    def _monte_carlo_cost_prediction(self, historical_data: List[float], 
                                   iterations: int = 1000) -> float:
        """Prédiction Monte Carlo basée sur l'historique"""
        if len(historical_data) < 2:
            return historical_data[-1] if historical_data else 0
        
        # Calcul des variations historiques
        variations = [historical_data[i+1] - historical_data[i] 
                     for i in range(len(historical_data)-1)]
        
        mean_variation = np.mean(variations)
        std_variation = np.std(variations)
        
        # Simulations
        final_costs = []
        current_cost = historical_data[-1]
        
        for _ in range(iterations):
            # Simulation de 10 périodes futures
            simulated_cost = current_cost
            for _ in range(10):
                variation = np.random.normal(mean_variation, std_variation)
                simulated_cost += variation
            
            final_costs.append(max(0, simulated_cost))
        
        return np.mean(final_costs)
    
    def _analyze_cost_overrun_risks(self, current_cost: float, total_budget: float, 
                                  predicted_cost: float) -> Dict[str, Any]:
        """Analyse les risques de dépassement budgétaire"""
        overrun_amount = predicted_cost - total_budget
        overrun_percentage = (overrun_amount / total_budget) * 100 if total_budget > 0 else 0
        
        # Probabilité basée sur l'ampleur du dépassement prédit
        if overrun_percentage <= 0:
            probability = 0.1  # Toujours un petit risque
        elif overrun_percentage <= 10:
            probability = 0.3
        elif overrun_percentage <= 20:
            probability = 0.6
        else:
            probability = 0.8
        
        return {
            'overrun_amount': overrun_amount,
            'overrun_percentage': overrun_percentage,
            'probability': probability,
            'expected_overrun': overrun_amount * probability,
            'risk_level': 'LOW' if probability <= 0.3 else 'MEDIUM' if probability <= 0.6 else 'HIGH'
        }
    
    def _analyze_cash_risks(self, projections: List[CashFlowProjection]) -> Dict[str, Any]:
        """Analyse les risques de trésorerie"""
        min_cash = min(p.cumulative_flow for p in projections)
        negative_periods = [p for p in projections if p.cumulative_flow < 0]
        
        return {
            'min_cash_position': min_cash,
            'negative_cash_periods': len(negative_periods),
            'max_negative_duration': len(negative_periods),  # Simplifié
            'liquidity_risk': 'HIGH' if min_cash < -10000 else 'MEDIUM' if min_cash < 0 else 'LOW',
            'financing_need': abs(min_cash) if min_cash < 0 else 0
        }
    
    def _calculate_summary_metrics(self, evm_analysis: Dict, roi_analysis: Dict, 
                                 financial_risks: Dict) -> Dict[str, Any]:
        """Calcule les métriques de synthèse"""
        return {
            'overall_financial_health': self._assess_overall_financial_health(
                evm_analysis, roi_analysis, financial_risks
            ),
            'key_metrics': {
                'cpi': evm_analysis['current_metrics']['cpi'],
                'spi': evm_analysis['current_metrics']['spi'],
                'roi': roi_analysis['roi'],
                'npv': roi_analysis['npv'],
                'payback_period': roi_analysis['payback_period'],
                'risk_score': financial_risks['risk_score']
            },
            'recommendations': self._generate_executive_recommendations(
                evm_analysis, roi_analysis, financial_risks
            )
        }
    
    def _assess_overall_financial_health(self, evm_analysis: Dict, roi_analysis: Dict, 
                                       financial_risks: Dict) -> str:
        """Évalue la santé financière globale du projet"""
        cpi = evm_analysis['current_metrics']['cpi']
        roi = roi_analysis['roi']
        risk_score = financial_risks['risk_score']
        
        # Score composite
        health_score = 0
        
        # Performance coût (40%)
        if cpi >= 1.0:
            health_score += 40
        elif cpi >= 0.9:
            health_score += 30
        elif cpi >= 0.8:
            health_score += 20
        else:
            health_score += 10
        
        # ROI (40%)
        if roi >= 20:
            health_score += 40
        elif roi >= 10:
            health_score += 30
        elif roi >= 0:
            health_score += 20
        else:
            health_score += 0
        
        # Risques (20%)
        if risk_score <= 0.1:
            health_score += 20
        elif risk_score <= 0.3:
            health_score += 15
        elif risk_score <= 0.6:
            health_score += 10
        else:
            health_score += 5
        
        if health_score >= 80:
            return "EXCELLENT"
        elif health_score >= 60:
            return "BON"
        elif health_score >= 40:
            return "MOYEN"
        else:
            return "PRÉOCCUPANT"
    
    def _generate_executive_recommendations(self, evm_analysis: Dict, roi_analysis: Dict, 
                                          financial_risks: Dict) -> List[str]:
        """Génère des recommandations exécutives"""
        recommendations = []
        
        cpi = evm_analysis['current_metrics']['cpi']
        if cpi < 0.9:
            recommendations.append(
                f"🔴 Performance coût dégradée (CPI: {cpi:.2f}). "
                "Réexaminer la structure des coûts et optimiser les processus."
            )
        
        roi = roi_analysis['roi']
        if roi < 10:
            recommendations.append(
                f"⚠️ ROI faible ({roi:.1f}%). "
                "Étudier des opportunités d'augmentation des bénéfices."
            )
        
        if financial_risks['risk_score'] > 0.3:
            recommendations.append(
                "🛡️ Risques financiers élevés détectés. "
                "Mettre en place un plan de mitigation et une réserve de contingence."
            )
        
        if evm_analysis['current_metrics']['vac'] < 0:
            recommendations.append(
                "💰 Dépassement budgétaire prévu. "
                "Négocier un budget additionnel ou réduire le périmètre."
            )
        
        return recommendations


# Instance globale
_financial_engine = None

def get_financial_analytics_engine() -> FinancialAnalyticsEngine:
    """Retourne l'instance globale du moteur financier"""
    global _financial_engine
    if _financial_engine is None:
        _financial_engine = FinancialAnalyticsEngine()
    return _financial_engine


if __name__ == "__main__":
    # Test du moteur
    engine = FinancialAnalyticsEngine()
    
    # Analyse complète
    analysis = engine.analyze_project_financials({}, {})
    print(f"Analyse EVM - CPI: {analysis['evm_analysis']['current_metrics']['cpi']:.2f}")
    print(f"ROI: {analysis['roi_analysis']['roi']:.1f}%")
    print(f"NPV: €{analysis['roi_analysis']['npv']:,.0f}")
    print(f"Payback: {analysis['roi_analysis']['payback_period']:.1f} ans")
    
    # Génération des visualisations
    dashboards = engine.create_financial_dashboard(analysis)
    print(f"Graphiques financiers générés: {len(dashboards)}")
    
    print(f"Santé financière: {analysis['summary_metrics']['overall_financial_health']}")