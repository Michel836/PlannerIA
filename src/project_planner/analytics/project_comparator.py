"""
Project Comparison Engine for PlannerIA Advanced AI Suite
Enables side-by-side analysis of multiple project scenarios
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Plotting imports
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)

class ProjectComparator:
    """Advanced project comparison and analytics engine"""
    
    def __init__(self):
        self.projects = {}
        self.comparison_metrics = [
            "total_duration", "total_cost", "total_effort", "risk_score",
            "complexity_score", "success_probability", "roi_estimate"
        ]
    
    def add_project(self, project_id: str, plan_data: Dict[str, Any], name: str = None) -> bool:
        """
        Add a project for comparison
        
        Args:
            project_id: Unique identifier for the project
            plan_data: Complete plan data from PlannerIA
            name: Optional display name
            
        Returns:
            Success status
        """
        try:
            if not plan_data or "error" in plan_data:
                logger.warning(f"Invalid plan data for project {project_id}")
                return False
            
            # Extract key metrics
            plan = plan_data.get("plan", {})
            overview = plan.get("project_overview", {})
            metrics = plan.get("project_metrics", {})
            risk_analysis = plan_data.get("risk_analysis", {})
            strategy_insights = plan_data.get("strategy_insights", {})
            
            project_summary = {
                "id": project_id,
                "name": name or f"Project {project_id}",
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "total_duration": overview.get("total_duration", 0),
                    "total_cost": overview.get("total_cost", 0),
                    "total_effort": metrics.get("total_estimated_effort", 0),
                    "task_count": metrics.get("task_count", 0),
                    "phase_count": len(plan.get("wbs", {}).get("phases", [])),
                    "risk_score": risk_analysis.get("risk_score", 0),
                    "high_risks": risk_analysis.get("high_risks", 0),
                    "complexity_score": metrics.get("complexity_score", 0),
                    "success_probability": strategy_insights.get("success_probability", 0),
                },
                "details": {
                    "title": overview.get("title", "Untitled Project"),
                    "description": overview.get("description", ""),
                    "objectives": overview.get("objectives", []),
                    "critical_path_duration": overview.get("critical_path_duration", 0),
                    "complexity_level": metrics.get("complexity_level", "Medium"),
                    "top_risks": risk_analysis.get("top_risks", [])[:3],
                },
                "full_data": plan_data  # Store complete data for detailed analysis
            }
            
            self.projects[project_id] = project_summary
            logger.info(f"Added project {project_id} for comparison")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add project {project_id}: {e}")
            return False
    
    def remove_project(self, project_id: str) -> bool:
        """Remove a project from comparison"""
        if project_id in self.projects:
            del self.projects[project_id]
            logger.info(f"Removed project {project_id} from comparison")
            return True
        return False
    
    def get_project_list(self) -> List[Dict[str, str]]:
        """Get list of available projects for comparison"""
        return [
            {
                "id": proj["id"],
                "name": proj["name"],
                "title": proj["details"]["title"],
                "timestamp": proj["timestamp"]
            }
            for proj in self.projects.values()
        ]
    
    def compare_projects(self, project_ids: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive comparison between selected projects
        
        Args:
            project_ids: List of project IDs to compare
            
        Returns:
            Comparison analysis with metrics, charts, and insights
        """
        if not project_ids or len(project_ids) < 2:
            return {"error": "At least 2 projects required for comparison"}
        
        # Filter valid projects
        valid_projects = [self.projects[pid] for pid in project_ids if pid in self.projects]
        
        if len(valid_projects) < 2:
            return {"error": "Not enough valid projects for comparison"}
        
        try:
            comparison = {
                "projects": valid_projects,
                "metrics_comparison": self._compare_metrics(valid_projects),
                "charts": self._generate_comparison_charts(valid_projects) if HAS_PLOTLY else None,
                "insights": self._generate_insights(valid_projects),
                "recommendations": self._generate_recommendations(valid_projects),
                "timestamp": datetime.now().isoformat()
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {"error": f"Comparison analysis failed: {e}"}
    
    def _compare_metrics(self, projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare key metrics across projects"""
        metrics_comparison = {}
        
        # Define metric configurations
        metric_configs = {
            "total_duration": {"unit": "jours", "format": "{:.0f}", "higher_better": False},
            "total_cost": {"unit": "€", "format": "{:,.0f}", "higher_better": False},
            "total_effort": {"unit": "h", "format": "{:.1f}", "higher_better": False},
            "risk_score": {"unit": "/10", "format": "{:.1f}", "higher_better": False},
            "success_probability": {"unit": "%", "format": "{:.0f}", "higher_better": True},
            "complexity_score": {"unit": "", "format": "{:.1f}", "higher_better": False},
            "task_count": {"unit": "tâches", "format": "{:.0f}", "higher_better": None},
            "phase_count": {"unit": "phases", "format": "{:.0f}", "higher_better": None}
        }
        
        for metric_name, config in metric_configs.items():
            values = []
            project_names = []
            
            for project in projects:
                value = project["metrics"].get(metric_name, 0)
                values.append(value)
                project_names.append(project["name"])
            
            if values and any(v > 0 for v in values):  # Only include metrics with data
                best_idx = (
                    values.index(max(values)) if config["higher_better"] 
                    else values.index(min(values)) if not config["higher_better"]
                    else None
                )
                
                metrics_comparison[metric_name] = {
                    "values": values,
                    "projects": project_names,
                    "unit": config["unit"],
                    "format": config["format"],
                    "best_project": project_names[best_idx] if best_idx is not None else None,
                    "difference": max(values) - min(values) if len(values) > 1 else 0
                }
        
        return metrics_comparison
    
    def _generate_comparison_charts(self, projects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison visualizations"""
        if not HAS_PLOTLY:
            return {}
        
        charts = {}
        
        try:
            # 1. Overview Radar Chart
            categories = ['Durée', 'Coût', 'Effort', 'Risque', 'Complexité']
            
            fig_radar = go.Figure()
            
            for project in projects:
                # Normalize values for radar chart (0-100 scale)
                values = [
                    max(0, 100 - (project["metrics"]["total_duration"] / 100 * 100)),  # Lower duration better
                    max(0, 100 - (project["metrics"]["total_cost"] / 50000 * 100)),   # Lower cost better  
                    max(0, 100 - (project["metrics"]["total_effort"] / 1000 * 100)), # Lower effort better
                    max(0, 100 - (project["metrics"]["risk_score"] * 10)),           # Lower risk better
                    max(0, 100 - (project["metrics"]["complexity_score"] * 10))      # Lower complexity better
                ]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=project["name"],
                    line=dict(width=2)
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100])
                ),
                title="Comparaison Multi-Critères",
                showlegend=True,
                height=400
            )
            
            charts["radar"] = fig_radar
            
            # 2. Cost vs Duration Scatter
            fig_scatter = go.Figure()
            
            for project in projects:
                fig_scatter.add_trace(go.Scatter(
                    x=[project["metrics"]["total_duration"]],
                    y=[project["metrics"]["total_cost"]],
                    mode='markers+text',
                    name=project["name"],
                    text=[project["name"]],
                    textposition="top center",
                    marker=dict(size=15, opacity=0.7),
                    hovertemplate=f"<b>{project['name']}</b><br>" +
                                f"Durée: {project['metrics']['total_duration']} jours<br>" +
                                f"Coût: €{project['metrics']['total_cost']:,.0f}<br>" +
                                f"Risque: {project['metrics']['risk_score']}/10<extra></extra>"
                ))
            
            fig_scatter.update_layout(
                title="Coût vs Durée",
                xaxis_title="Durée (jours)",
                yaxis_title="Coût (€)",
                showlegend=False,
                height=400
            )
            
            charts["scatter"] = fig_scatter
            
            # 3. Risk Comparison Bar Chart
            risk_data = [
                {
                    "name": project["name"],
                    "risk_score": project["metrics"]["risk_score"],
                    "high_risks": project["metrics"]["high_risks"]
                }
                for project in projects
            ]
            
            fig_risk = go.Figure(data=[
                go.Bar(
                    name='Score de Risque',
                    x=[p["name"] for p in risk_data],
                    y=[p["risk_score"] for p in risk_data],
                    yaxis='y',
                    offsetgroup=1
                ),
                go.Bar(
                    name='Risques Élevés',
                    x=[p["name"] for p in risk_data],
                    y=[p["high_risks"] for p in risk_data],
                    yaxis='y2',
                    offsetgroup=2
                )
            ])
            
            fig_risk.update_layout(
                title="Analyse Comparative des Risques",
                xaxis_title="Projets",
                yaxis=dict(title="Score de Risque (/10)", side="left"),
                yaxis2=dict(title="Nombre de Risques Élevés", side="right", overlaying="y"),
                barmode='group',
                height=400
            )
            
            charts["risk_comparison"] = fig_risk
            
        except Exception as e:
            logger.error(f"Chart generation failed: {e}")
            charts["error"] = f"Chart generation failed: {e}"
        
        return charts
    
    def _generate_insights(self, projects: List[Dict[str, Any]]) -> List[str]:
        """Generate intelligent insights from project comparison"""
        insights = []
        
        try:
            # Cost efficiency analysis
            costs = [p["metrics"]["total_cost"] for p in projects]
            durations = [p["metrics"]["total_duration"] for p in projects]
            
            if costs and durations:
                cost_efficiency = [(c/d if d > 0 else 0) for c, d in zip(costs, durations)]
                best_efficiency_idx = cost_efficiency.index(min([ce for ce in cost_efficiency if ce > 0]))
                
                insights.append(
                    f"💰 **Efficacité coût**: {projects[best_efficiency_idx]['name']} offre le meilleur rapport coût/durée "
                    f"({cost_efficiency[best_efficiency_idx]:,.0f}€/jour)"
                )
            
            # Risk analysis
            risk_scores = [p["metrics"]["risk_score"] for p in projects]
            if risk_scores:
                lowest_risk_idx = risk_scores.index(min(risk_scores))
                highest_risk_idx = risk_scores.index(max(risk_scores))
                
                insights.append(
                    f"⚠️ **Analyse des risques**: {projects[lowest_risk_idx]['name']} présente le risque le plus faible "
                    f"({risk_scores[lowest_risk_idx]:.1f}/10), tandis que {projects[highest_risk_idx]['name']} "
                    f"nécessite une attention particulière ({risk_scores[highest_risk_idx]:.1f}/10)"
                )
            
            # Success probability
            success_probs = [p["metrics"]["success_probability"] for p in projects]
            if success_probs and any(sp > 0 for sp in success_probs):
                best_success_idx = success_probs.index(max(success_probs))
                insights.append(
                    f"🎯 **Probabilité de succès**: {projects[best_success_idx]['name']} a la plus haute "
                    f"probabilité de succès ({success_probs[best_success_idx]:.0f}%)"
                )
            
            # Complexity analysis
            complexities = [p["metrics"]["complexity_score"] for p in projects]
            if complexities:
                simplest_idx = complexities.index(min(complexities))
                most_complex_idx = complexities.index(max(complexities))
                
                insights.append(
                    f"🧩 **Complexité**: {projects[simplest_idx]['name']} est le plus simple à implémenter "
                    f"(score: {complexities[simplest_idx]:.1f}), {projects[most_complex_idx]['name']} "
                    f"est le plus complexe (score: {complexities[most_complex_idx]:.1f})"
                )
            
            # Overall recommendation
            if len(projects) == 2:
                insights.append(self._generate_head_to_head_insight(projects))
            
        except Exception as e:
            logger.error(f"Insight generation failed: {e}")
            insights.append(f"⚠️ Impossible de générer des insights: {e}")
        
        return insights
    
    def _generate_head_to_head_insight(self, projects: List[Dict[str, Any]]) -> str:
        """Generate head-to-head comparison insight for 2 projects"""
        p1, p2 = projects[0], projects[1]
        
        p1_advantages = []
        p2_advantages = []
        
        # Compare key metrics
        comparisons = [
            ("durée", "total_duration", False),  # Lower is better
            ("coût", "total_cost", False),
            ("risque", "risk_score", False),
            ("probabilité de succès", "success_probability", True)  # Higher is better
        ]
        
        for metric_name, key, higher_better in comparisons:
            v1 = p1["metrics"].get(key, 0)
            v2 = p2["metrics"].get(key, 0)
            
            if v1 != v2:
                if (v1 > v2 and higher_better) or (v1 < v2 and not higher_better):
                    p1_advantages.append(metric_name)
                else:
                    p2_advantages.append(metric_name)
        
        if p1_advantages and p2_advantages:
            return (f"⚖️ **Comparaison directe**: {p1['name']} excelle en {', '.join(p1_advantages)}, "
                   f"tandis que {p2['name']} est supérieur en {', '.join(p2_advantages)}")
        elif p1_advantages:
            return f"🏆 **Avantage**: {p1['name']} surpasse {p2['name']} dans tous les domaines clés"
        elif p2_advantages:
            return f"🏆 **Avantage**: {p2['name']} surpasse {p1['name']} dans tous les domaines clés"
        else:
            return f"🤝 **Équilibré**: {p1['name']} et {p2['name']} présentent des caractéristiques similaires"
    
    def _generate_recommendations(self, projects: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable recommendations based on comparison"""
        recommendations = []
        
        try:
            # Find extremes
            costs = [(p["metrics"]["total_cost"], p["name"]) for p in projects]
            durations = [(p["metrics"]["total_duration"], p["name"]) for p in projects]
            risks = [(p["metrics"]["risk_score"], p["name"]) for p in projects]
            
            lowest_cost = min(costs, key=lambda x: x[0])
            shortest_duration = min(durations, key=lambda x: x[0])
            lowest_risk = min(risks, key=lambda x: x[0])
            
            recommendations.append(
                f"💡 **Pour un budget limité**: Choisissez {lowest_cost[1]} (€{lowest_cost[0]:,.0f})"
            )
            
            recommendations.append(
                f"⏱️ **Pour une livraison rapide**: Optez pour {shortest_duration[1]} ({shortest_duration[0]:.0f} jours)"
            )
            
            recommendations.append(
                f"🛡️ **Pour minimiser les risques**: Préférez {lowest_risk[1]} (risque: {lowest_risk[0]:.1f}/10)"
            )
            
            # Strategic recommendation
            if len(projects) >= 2:
                # Calculate overall score (weighted)
                scores = []
                for project in projects:
                    score = (
                        (100 - min(project["metrics"]["total_cost"] / 1000, 100)) * 0.3 +  # Cost (30%)
                        (100 - min(project["metrics"]["total_duration"] / 10, 100)) * 0.25 +  # Duration (25%)
                        (100 - project["metrics"]["risk_score"] * 10) * 0.25 +               # Risk (25%)
                        project["metrics"]["success_probability"] * 0.2                      # Success (20%)
                    )
                    scores.append((score, project["name"]))
                
                best_overall = max(scores, key=lambda x: x[0])
                recommendations.append(
                    f"🎯 **Recommandation globale**: {best_overall[1]} présente le meilleur équilibre "
                    f"global (score: {best_overall[0]:.0f}/100)"
                )
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            recommendations.append("⚠️ Impossible de générer des recommandations détaillées")
        
        return recommendations
    
    def save_comparison(self, comparison_data: Dict[str, Any], filename: str = None) -> str:
        """Save comparison results to file"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"comparison_{timestamp}.json"
            
            # Ensure comparisons directory exists
            comparisons_dir = os.path.join("data", "comparisons")
            os.makedirs(comparisons_dir, exist_ok=True)
            
            filepath = os.path.join(comparisons_dir, filename)
            
            # Remove non-serializable chart objects before saving
            save_data = comparison_data.copy()
            if "charts" in save_data:
                save_data["charts"] = "Charts generated but not serializable"
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Comparison saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save comparison: {e}")
            raise
    
    def load_comparison(self, filepath: str) -> Dict[str, Any]:
        """Load comparison results from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
            
            logger.info(f"Comparison loaded from {filepath}")
            return comparison_data
            
        except Exception as e:
            logger.error(f"Failed to load comparison: {e}")
            raise


# Convenience functions
def create_comparator() -> ProjectComparator:
    """Create a new project comparator instance"""
    return ProjectComparator()

def quick_compare(project_data_list: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, Any]:
    """
    Quick comparison of multiple projects
    
    Args:
        project_data_list: List of (project_id, plan_data, name) tuples
        
    Returns:
        Comparison results
    """
    comparator = ProjectComparator()
    
    project_ids = []
    for project_id, plan_data, name in project_data_list:
        if comparator.add_project(project_id, plan_data, name):
            project_ids.append(project_id)
    
    if len(project_ids) < 2:
        return {"error": "Not enough valid projects for comparison"}
    
    return comparator.compare_projects(project_ids)