"""
What-If Scenario Simulator for PlannerIA Dashboard
Interactive scenario testing and impact analysis for project planning.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import copy


def render_whatif_simulator(plan_data: Dict[str, Any]):
    """Main what-if simulator interface"""
    
    st.header("What-If Scenario Simulator")
    st.markdown("Test different scenarios and see their impact on project timeline, budget, and risks.")
    
    # Initialize session state for scenario data
    if 'scenario_data' not in st.session_state:
        st.session_state.scenario_data = initialize_scenario_data(plan_data)
    
    # Scenario controls
    render_scenario_controls(plan_data)
    
    # Impact analysis
    render_impact_analysis(plan_data)
    
    # Comparison dashboard
    render_scenario_comparison(plan_data)


def initialize_scenario_data(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Initialize baseline scenario data"""
    
    overview = plan_data.get('project_overview', {})
    
    return {
        'baseline': {
            'duration': overview.get('total_duration', 0),
            'cost': overview.get('total_cost', 0),
            'team_size': len(plan_data.get('resources', [])),
            'risk_level': calculate_overall_risk_level(plan_data)
        },
        'current': {
            'duration_multiplier': 1.0,
            'cost_multiplier': 1.0,
            'team_multiplier': 1.0,
            'complexity_factor': 1.0,
            'risk_tolerance': 0.5
        }
    }


def render_scenario_controls(plan_data: Dict[str, Any]):
    """Render interactive scenario adjustment controls"""
    
    st.subheader("Scenario Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Timeline & Resources**")
        
        duration_multiplier = st.slider(
            "Duration Adjustment",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.scenario_data['current']['duration_multiplier'],
            step=0.1,
            format="%.1fx",
            help="Adjust overall project duration (1.0 = baseline)"
        )
        
        team_multiplier = st.slider(
            "Team Size Multiplier",
            min_value=0.5,
            max_value=3.0,
            value=st.session_state.scenario_data['current']['team_multiplier'],
            step=0.1,
            format="%.1fx",
            help="Scale team size up or down"
        )
        
        complexity_factor = st.slider(
            "Project Complexity",
            min_value=0.5,
            max_value=2.0,
            value=st.session_state.scenario_data['current']['complexity_factor'],
            step=0.1,
            format="%.1fx",
            help="Adjust for project complexity changes"
        )
    
    with col2:
        st.markdown("**Budget & Risk**")
        
        cost_multiplier = st.slider(
            "Budget Adjustment",
            min_value=0.5,
            max_value=2.5,
            value=st.session_state.scenario_data['current']['cost_multiplier'],
            step=0.1,
            format="%.1fx",
            help="Adjust budget allocation"
        )
        
        risk_tolerance = st.slider(
            "Risk Tolerance",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.scenario_data['current']['risk_tolerance'],
            step=0.1,
            format="%.1f",
            help="Organization's risk tolerance (0=risk averse, 1=risk taking)"
        )
        
        # Advanced scenario presets
        st.markdown("**Quick Scenarios**")
        scenario_preset = st.selectbox(
            "Apply Preset Scenario",
            options=[
                "Current Settings",
                "Aggressive Timeline",
                "Budget Constrained", 
                "Resource Limited",
                "High Quality Focus",
                "Crisis Mode"
            ],
            help="Apply predefined scenario configurations"
        )
    
    # Update session state
    st.session_state.scenario_data['current'].update({
        'duration_multiplier': duration_multiplier,
        'cost_multiplier': cost_multiplier,
        'team_multiplier': team_multiplier,
        'complexity_factor': complexity_factor,
        'risk_tolerance': risk_tolerance
    })
    
    # Apply preset if changed
    if scenario_preset != "Current Settings":
        apply_scenario_preset(scenario_preset)


def apply_scenario_preset(preset_name: str):
    """Apply predefined scenario configurations"""
    
    presets = {
        "Aggressive Timeline": {
            'duration_multiplier': 0.7,
            'cost_multiplier': 1.2,
            'team_multiplier': 1.5,
            'complexity_factor': 1.3,
            'risk_tolerance': 0.8
        },
        "Budget Constrained": {
            'duration_multiplier': 1.3,
            'cost_multiplier': 0.8,
            'team_multiplier': 0.8,
            'complexity_factor': 0.9,
            'risk_tolerance': 0.3
        },
        "Resource Limited": {
            'duration_multiplier': 1.5,
            'cost_multiplier': 1.1,
            'team_multiplier': 0.6,
            'complexity_factor': 1.2,
            'risk_tolerance': 0.4
        },
        "High Quality Focus": {
            'duration_multiplier': 1.2,
            'cost_multiplier': 1.3,
            'team_multiplier': 1.1,
            'complexity_factor': 0.8,
            'risk_tolerance': 0.2
        },
        "Crisis Mode": {
            'duration_multiplier': 0.6,
            'cost_multiplier': 1.8,
            'team_multiplier': 2.0,
            'complexity_factor': 1.8,
            'risk_tolerance': 0.9
        }
    }
    
    if preset_name in presets:
        st.session_state.scenario_data['current'].update(presets[preset_name])
        st.experimental_rerun()


def render_impact_analysis(plan_data: Dict[str, Any]):
    """Calculate and display scenario impact analysis"""
    
    st.subheader("Impact Analysis")
    
    # Calculate scenario impacts
    baseline = st.session_state.scenario_data['baseline']
    current = st.session_state.scenario_data['current']
    
    # Complex impact calculations
    impacts = calculate_scenario_impacts(baseline, current, plan_data)
    
    # Display impact metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        duration_change = impacts['duration'] - baseline['duration']
        st.metric(
            "Project Duration",
            f"{impacts['duration']:.1f} days",
            f"{duration_change:+.1f} days",
            delta_color="inverse" if duration_change > 0 else "normal"
        )
    
    with col2:
        cost_change = impacts['cost'] - baseline['cost']
        st.metric(
            "Total Cost", 
            f"${impacts['cost']:,.0f}",
            f"${cost_change:+,.0f}",
            delta_color="inverse" if cost_change > 0 else "normal"
        )
    
    with col3:
        risk_change = impacts['risk_score'] - baseline['risk_level']
        risk_level = get_risk_level_text(impacts['risk_score'])
        st.metric(
            "Risk Level",
            risk_level,
            f"{risk_change:+.1f}",
            delta_color="inverse" if risk_change > 0 else "normal"
        )
    
    with col4:
        success_prob = impacts['success_probability']
        st.metric(
            "Success Probability",
            f"{success_prob:.1%}",
            help="Estimated probability of project success under this scenario"
        )
    
    # Impact visualization
    render_impact_charts(baseline, impacts)
    
    # Risk assessment
    render_scenario_risks(impacts, plan_data)


def calculate_scenario_impacts(baseline: Dict[str, Any], 
                             scenario: Dict[str, Any],
                             plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate complex scenario impacts using project management formulas"""
    
    # Base values
    base_duration = baseline['duration']
    base_cost = baseline['cost']
    base_team = baseline['team_size']
    
    # Scenario multipliers
    duration_mult = scenario['duration_multiplier']
    cost_mult = scenario['cost_multiplier'] 
    team_mult = scenario['team_multiplier']
    complexity = scenario['complexity_factor']
    risk_tolerance = scenario['risk_tolerance']
    
    # Brooks' Law effect: Adding people to late projects makes them later
    if team_mult > 1.2:
        communication_overhead = 1 + (team_mult - 1) * 0.15
        duration_mult *= communication_overhead
    
    # Economy of scale for larger teams (up to a point)
    if team_mult > 1.0 and team_mult <= 1.5:
        efficiency_gain = 0.9 + (0.1 / team_mult if team_mult > 0 else 0.1)
        duration_mult *= efficiency_gain
    
    # Complexity impact on duration and cost
    complexity_duration_impact = complexity ** 0.8
    complexity_cost_impact = complexity ** 1.2
    
    # Calculate new values
    new_duration = base_duration * duration_mult * complexity_duration_impact
    new_cost = base_cost * cost_mult * team_mult * complexity_cost_impact
    
    # Risk calculations
    risk_factors = []
    
    # Timeline pressure risk
    if duration_mult < 0.8:
        risk_factors.append(("Timeline Pressure", (0.8 - duration_mult) * 10))
    
    # Resource risk
    if team_mult < 0.7:
        risk_factors.append(("Resource Shortage", (0.7 - team_mult) * 8))
    elif team_mult > 1.5:
        risk_factors.append(("Team Coordination", (team_mult - 1.5) * 6))
    
    # Budget risk
    if cost_mult < 0.8:
        risk_factors.append(("Budget Constraints", (0.8 - cost_mult) * 7))
    
    # Complexity risk
    if complexity > 1.2:
        risk_factors.append(("High Complexity", (complexity - 1.2) * 5))
    
    total_risk_score = sum(score for _, score in risk_factors)
    adjusted_risk = baseline['risk_level'] + total_risk_score * (1 - risk_tolerance)
    
    # Success probability calculation
    success_factors = {
        'timeline_factor': max(0.3, 1 - abs(duration_mult - 1) * 0.5),
        'budget_factor': max(0.4, 1 - abs(cost_mult - 1) * 0.3), 
        'team_factor': max(0.5, 1 - abs(team_mult - 1) * 0.4),
        'complexity_factor': max(0.3, 1 - (complexity - 1) * 0.6),
        'risk_factor': max(0.2, 1 - (adjusted_risk - baseline['risk_level']) * 0.1)
    }
    
    success_probability = np.prod(list(success_factors.values()))
    
    return {
        'duration': new_duration,
        'cost': new_cost,
        'team_size': base_team * team_mult,
        'risk_score': adjusted_risk,
        'success_probability': success_probability,
        'risk_factors': risk_factors,
        'success_factors': success_factors
    }


def render_impact_charts(baseline: Dict[str, Any], impacts: Dict[str, Any]):
    """Render impact visualization charts"""
    
    st.subheader("Visual Impact Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for multiple dimensions
        categories = ['Duration', 'Cost', 'Team Size', 'Risk Level']
        
        # Normalize values for radar chart (baseline = 1.0) with zero division protection
        baseline_values = [1.0, 1.0, 1.0, 1.0]
        scenario_values = [
            impacts['duration'] / baseline['duration'] if baseline['duration'] > 0 else 1.0,
            impacts['cost'] / baseline['cost'] if baseline['cost'] > 0 else 1.0,
            impacts['team_size'] / baseline['team_size'] if baseline['team_size'] > 0 else 1.0,
            impacts['risk_score'] / baseline['risk_level'] if baseline['risk_level'] > 0 else 1.0
        ]
        
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=baseline_values,
            theta=categories,
            fill='toself',
            name='Baseline',
            line_color='blue',
            fillcolor='rgba(0,0,255,0.1)'
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=scenario_values,
            theta=categories, 
            fill='toself',
            name='Scenario',
            line_color='red',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 2.5]
                )
            ),
            showlegend=True,
            title="Scenario vs Baseline Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Success factors breakdown
        success_factors = impacts['success_factors']
        
        factor_df = pd.DataFrame([
            {'Factor': factor.replace('_', ' ').title(), 'Score': score}
            for factor, score in success_factors.items()
        ])
        
        fig_factors = px.bar(
            factor_df,
            x='Score',
            y='Factor',
            orientation='h',
            title="Success Factor Analysis",
            color='Score',
            color_continuous_scale='RdYlGn',
            range_color=[0, 1]
        )
        
        fig_factors.update_layout(height=400)
        st.plotly_chart(fig_factors, use_container_width=True)


def render_scenario_risks(impacts: Dict[str, Any], plan_data: Dict[str, Any]):
    """Display scenario-specific risks and recommendations"""
    
    st.subheader("Scenario Risk Assessment")
    
    risk_factors = impacts.get('risk_factors', [])
    
    if risk_factors:
        st.markdown("**New Risk Factors Introduced:**")
        
        for risk_name, risk_score in risk_factors:
            risk_level = "High" if risk_score > 7 else "Medium" if risk_score > 3 else "Low"
            color = "🔴" if risk_level == "High" else "🟡" if risk_level == "Medium" else "🟢"
            
            st.markdown(f"• {color} **{risk_name}**: {risk_level} (Score: {risk_score:.1f})")
    
    # Recommendations based on scenario
    recommendations = generate_scenario_recommendations(impacts, plan_data)
    
    if recommendations:
        st.markdown("**Recommendations:**")
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")


def render_scenario_comparison(plan_data: Dict[str, Any]):
    """Compare multiple scenarios side by side"""
    
    st.subheader("Scenario Comparison")
    
    # Save current scenario
    if st.button("Save Current Scenario"):
        scenario_name = st.text_input("Scenario Name", value=f"Scenario {len(st.session_state.get('saved_scenarios', [])) + 1}")
        if scenario_name:
            save_scenario(scenario_name)
            st.success(f"Scenario '{scenario_name}' saved!")
    
    # Display saved scenarios
    saved_scenarios = st.session_state.get('saved_scenarios', [])
    
    if saved_scenarios:
        # Create comparison table
        comparison_data = []
        baseline = st.session_state.scenario_data['baseline']
        
        # Add baseline
        comparison_data.append({
            'Scenario': 'Baseline',
            'Duration (days)': baseline['duration'],
            'Cost ($)': baseline['cost'],
            'Team Size': baseline['team_size'],
            'Risk Level': baseline['risk_level'],
            'Success Prob': 0.75  # Default baseline probability
        })
        
        # Add saved scenarios
        for scenario in saved_scenarios:
            impacts = calculate_scenario_impacts(baseline, scenario['parameters'], plan_data)
            comparison_data.append({
                'Scenario': scenario['name'],
                'Duration (days)': impacts['duration'],
                'Cost ($)': impacts['cost'],
                'Team Size': impacts['team_size'], 
                'Risk Level': impacts['risk_score'],
                'Success Prob': impacts['success_probability']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Format the dataframe for display
        st.dataframe(
            df_comparison.style.format({
                'Duration (days)': '{:.1f}',
                'Cost ($)': '${:,.0f}',
                'Team Size': '{:.1f}',
                'Risk Level': '{:.1f}',
                'Success Prob': '{:.1%}'
            }),
            use_container_width=True
        )
        
        # Comparison chart
        fig_comparison = px.scatter(
            df_comparison,
            x='Duration (days)',
            y='Cost ($)',
            size='Success Prob',
            color='Risk Level',
            hover_data=['Team Size'],
            title="Scenario Comparison: Duration vs Cost"
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Clear scenarios button
        if st.button("Clear All Scenarios"):
            st.session_state.saved_scenarios = []
            st.experimental_rerun()


def save_scenario(name: str):
    """Save current scenario configuration"""
    if 'saved_scenarios' not in st.session_state:
        st.session_state.saved_scenarios = []
    
    scenario = {
        'name': name,
        'parameters': copy.deepcopy(st.session_state.scenario_data['current']),
        'timestamp': datetime.now().isoformat()
    }
    
    st.session_state.saved_scenarios.append(scenario)


def generate_scenario_recommendations(impacts: Dict[str, Any], plan_data: Dict[str, Any]) -> List[str]:
    """Generate context-aware recommendations based on scenario"""
    
    recommendations = []
    success_prob = impacts['success_probability']
    
    if success_prob < 0.5:
        recommendations.append("Consider revising scenario parameters - current success probability is low")
    
    if impacts['duration'] > st.session_state.scenario_data['baseline']['duration'] * 1.3:
        recommendations.append("Timeline extension is significant - evaluate if stakeholders will accept delay")
    
    if impacts['cost'] > st.session_state.scenario_data['baseline']['cost'] * 1.2:
        recommendations.append("Budget increase requires approval - prepare business case for additional funding")
    
    risk_factors = impacts.get('risk_factors', [])
    high_risks = [rf for rf in risk_factors if rf[1] > 7]
    
    if high_risks:
        recommendations.append("High-risk factors detected - develop specific mitigation strategies")
    
    if impacts['team_size'] > st.session_state.scenario_data['baseline']['team_size'] * 1.5:
        recommendations.append("Large team increase - plan for communication overhead and coordination challenges")
    
    return recommendations


def calculate_overall_risk_level(plan_data: Dict[str, Any]) -> float:
    """Calculate overall project risk level from risk data"""
    
    risks = plan_data.get('risks', [])
    if not risks:
        return 5.0  # Default moderate risk
    
    risk_scores = [r.get('risk_score', 0) for r in risks]
    return np.mean(risk_scores) if risk_scores else 5.0


def get_risk_level_text(risk_score: float) -> str:
    """Convert risk score to descriptive text"""
    
    if risk_score >= 15:
        return "High"
    elif risk_score >= 9:
        return "Medium"
    elif risk_score >= 5:
        return "Low-Medium"
    else:
        return "Low"