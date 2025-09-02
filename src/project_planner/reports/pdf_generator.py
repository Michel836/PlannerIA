"""
PDF Report Generator for PlannerIA
Generates comprehensive PDF reports from project plans
"""

import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

# ReportLab imports
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.platypus import PageBreak, Image, KeepTogether
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

# Optional plotting imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Configure logging
logger = logging.getLogger(__name__)

class PDFReportGenerator:
    """Generate professional PDF reports from PlannerIA project plans"""
    
    def __init__(self):
        if not HAS_REPORTLAB:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86C1'),
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.HexColor('#5DADE2'),
            alignment=TA_LEFT
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#1B4F72'),
            alignment=TA_LEFT
        ))
        
        # Highlight box
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            backColor=colors.HexColor('#EBF5FB'),
            borderColor=colors.HexColor('#5DADE2'),
            borderWidth=1,
            leftIndent=10,
            rightIndent=10,
            topPadding=8,
            bottomPadding=8
        ))
    
    def generate_report(self, plan_data: Dict[str, Any], output_path: Optional[str] = None, include_charts: bool = True) -> str:
        """
        Generate comprehensive PDF report from plan data
        
        Args:
            plan_data: Complete plan data with all insights
            output_path: Optional output path, if None will generate in data/reports/
            include_charts: Whether to include charts and visualizations
            
        Returns:
            Path to generated PDF file
        """
        if not plan_data or "error" in plan_data:
            raise ValueError("Invalid plan data provided for PDF generation")
        
        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = plan_data.get("plan", {}).get("project_overview", {}).get("title", "project")
            project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
            filename = f"PlannerIA_Report_{project_name}_{timestamp}.pdf"
            
            # Ensure reports directory exists
            reports_dir = os.path.join("data", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            output_path = os.path.join(reports_dir, filename)
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            # Build content
            story = []
            story.extend(self._build_title_page(plan_data))
            story.extend(self._build_executive_summary(plan_data))
            story.extend(self._build_project_overview(plan_data))
            
            # Add Gantt chart if available
            if include_charts and HAS_MATPLOTLIB:
                gantt_elements = self._build_gantt_chart(plan_data)
                if gantt_elements:
                    story.extend(gantt_elements)
            
            story.extend(self._build_wbs_section(plan_data))
            
            # Add budget breakdown chart if available
            if include_charts and HAS_MATPLOTLIB:
                budget_elements = self._build_budget_chart(plan_data)
                if budget_elements:
                    story.extend(budget_elements)
            
            story.extend(self._build_risk_analysis(plan_data))
            
            # Add risk matrix if available
            if include_charts and HAS_MATPLOTLIB:
                risk_matrix = self._build_risk_matrix(plan_data)
                if risk_matrix:
                    story.extend(risk_matrix)
            
            story.extend(self._build_ai_insights(plan_data))
            story.extend(self._build_critical_path_analysis(plan_data))
            story.extend(self._build_resource_allocation(plan_data))
            story.extend(self._build_quality_metrics(plan_data))
            story.extend(self._build_recommendations(plan_data))
            story.extend(self._build_appendix(plan_data))
            
            # Generate PDF
            doc.build(story)
            
            logger.info(f"PDF report generated successfully: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise
    
    def _build_title_page(self, plan_data: Dict[str, Any]) -> List:
        """Build title page elements"""
        story = []
        plan = plan_data.get("plan", {})
        overview = plan.get("project_overview", {})
        
        # Main title
        title = overview.get("title", "Project Plan")
        story.append(Paragraph(title, self.styles['CustomTitle']))
        story.append(Spacer(1, 20))
        
        # Subtitle
        story.append(Paragraph("Intelligence Artificielle • Planification de Projet", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 40))
        
        # Format durations and costs properly
        total_duration = overview.get('total_duration', 0)
        if isinstance(total_duration, (int, float)):
            duration_str = f"{total_duration:.0f} jours"
        else:
            duration_str = f"{total_duration} jours"
        
        total_cost = overview.get('total_cost', 0)
        if isinstance(total_cost, (int, float)):
            cost_str = f"€{total_cost:,.0f}"
        else:
            cost_str = str(total_cost)
        
        critical_path = overview.get('critical_path_duration', 0)
        if isinstance(critical_path, (int, float)):
            critical_str = f"{critical_path:.0f} jours"
        else:
            critical_str = f"{critical_path} jours"
        
        # Enhanced metrics table with more information
        metrics_data = [
            ['Métrique', 'Valeur', 'Statut'],
            ['Durée Totale', duration_str, '✓'],
            ['Budget Total', cost_str, '✓'],
            ['Chemin Critique', critical_str, '✓'],
            ['Nombre de Phases', str(len(plan.get("wbs", {}).get("phases", []))), '✓'],
            ['Nombre de Tâches', str(sum(len(p.get('tasks', [])) for p in plan.get("wbs", {}).get("phases", []))), '✓'],
            ['Score de Risque', f"{plan_data.get('risk_analysis', {}).get('risk_score', 0)}/10", '⚠' if plan_data.get('risk_analysis', {}).get('risk_score', 0) > 6 else '✓']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.8*inch, 0.5*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5DADE2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF5FB')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#5DADE2'))
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 40))
        
        # Generation info
        gen_date = datetime.now().strftime("%d/%m/%Y à %H:%M")
        story.append(Paragraph(f"Rapport généré le {gen_date}", self.styles['Normal']))
        story.append(Paragraph("Propulsé par PlannerIA - Intelligence Artificielle Multi-Agents", self.styles['Normal']))
        story.append(Paragraph(f"Version: 2.0 | 20 Systèmes IA Actifs", self.styles['Normal']))
        
        story.append(PageBreak())
        return story
    
    def _build_executive_summary(self, plan_data: Dict[str, Any]) -> List:
        """Build executive summary section"""
        story = []
        plan = plan_data.get("plan", {})
        
        story.append(Paragraph("Résumé Exécutif", self.styles['CustomSubtitle']))
        
        # Project description
        description = plan.get("project_overview", {}).get("description", "")
        if description:
            story.append(Paragraph("Description du Projet", self.styles['SectionHeader']))
            story.append(Paragraph(description, self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Key highlights
        overview = plan.get("project_overview", {})
        highlights = [
            f"• Projet planifié sur {overview.get('total_duration', 'N/A')} jours",
            f"• Budget estimé: €{overview.get('total_cost', 0):,.0f}",
            f"• Chemin critique: {overview.get('critical_path_duration', 'N/A')} jours",
            f"• {len(plan.get('wbs', {}).get('phases', []))} phases identifiées"
        ]
        
        story.append(Paragraph("Points Clés", self.styles['SectionHeader']))
        for highlight in highlights:
            story.append(Paragraph(highlight, self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_project_overview(self, plan_data: Dict[str, Any]) -> List:
        """Build detailed project overview section"""
        story = []
        plan = plan_data.get("plan", {})
        overview = plan.get("project_overview", {})
        
        story.append(Paragraph("Vue d'Ensemble du Projet", self.styles['CustomSubtitle']))
        
        # Objectives
        objectives = overview.get("objectives", [])
        if objectives and isinstance(objectives, list):
            story.append(Paragraph("Objectifs", self.styles['SectionHeader']))
            for i, obj in enumerate(objectives[:5], 1):  # Limit to 5 objectives
                story.append(Paragraph(f"{i}. {obj}", self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Scope
        scope = overview.get("scope", "")
        if scope:
            story.append(Paragraph("Périmètre", self.styles['SectionHeader']))
            story.append(Paragraph(scope, self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        # Constraints
        constraints = overview.get("constraints", [])
        if constraints and isinstance(constraints, list):
            story.append(Paragraph("Contraintes", self.styles['SectionHeader']))
            for constraint in constraints[:3]:  # Limit to 3 constraints
                story.append(Paragraph(f"• {constraint}", self.styles['Normal']))
            story.append(Spacer(1, 12))
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_wbs_section(self, plan_data: Dict[str, Any]) -> List:
        """Build Work Breakdown Structure section"""
        story = []
        plan = plan_data.get("plan", {})
        wbs = plan.get("wbs", {})
        
        if not wbs:
            return story
        
        story.append(Paragraph("Structure de Découpage du Travail (WBS)", self.styles['CustomSubtitle']))
        
        phases = wbs.get("phases", [])
        if not phases:
            return story
        
        for phase_num, phase in enumerate(phases[:10], 1):  # Limit to 10 phases
            if not isinstance(phase, dict):
                continue
                
            phase_name = phase.get("name", f"Phase {phase_num}")
            story.append(Paragraph(f"Phase {phase_num}: {phase_name}", self.styles['SectionHeader']))
            
            # Phase description
            description = phase.get("description", "")
            if description:
                story.append(Paragraph(description, self.styles['Normal']))
            
            # Tasks table
            tasks = phase.get("tasks", [])
            if tasks and isinstance(tasks, list):
                task_data = [['Tâche', 'Durée (j)', 'Coût (€)', 'Ressources']]
                
                for task in tasks[:8]:  # Limit to 8 tasks per phase
                    if isinstance(task, dict):
                        task_name = task.get("name", "Tâche sans nom")[:40]  # Truncate long names
                        duration = task.get("duration", "N/A")
                        cost = f"€{task.get('cost', 0):,.0f}" if task.get('cost') else "N/A"
                        resources = ", ".join(task.get("assigned_resources", [])[:2])  # Max 2 resources
                        if not resources:
                            resources = "Non assigné"
                        
                        task_data.append([task_name, str(duration), cost, resources])
                
                if len(task_data) > 1:  # Only create table if there are tasks
                    task_table = Table(task_data, colWidths=[2.5*inch, 0.8*inch, 0.8*inch, 1.4*inch])
                    task_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#85C1E9')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('ALIGN', (1, 1), (2, -1), 'RIGHT'),  # Align numbers to right
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F8FF')),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#85C1E9')),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP')
                    ]))
                    
                    story.append(Spacer(1, 8))
                    story.append(task_table)
            
            story.append(Spacer(1, 16))
        
        return story
    
    def _build_risk_analysis(self, plan_data: Dict[str, Any]) -> List:
        """Build risk analysis section"""
        story = []
        risk_analysis = plan_data.get("risk_analysis")
        
        if not risk_analysis:
            return story
        
        story.append(Paragraph("Analyse des Risques", self.styles['CustomSubtitle']))
        
        # Risk metrics
        risk_score = risk_analysis.get("risk_score", 0)
        high_risks = risk_analysis.get("high_risks", 0)
        medium_risks = risk_analysis.get("medium_risks", 0)
        
        risk_summary = f"""
        Score Global de Risque: {risk_score}/10<br/>
        Risques Élevés: {high_risks}<br/>
        Risques Moyens: {medium_risks}
        """
        
        story.append(Paragraph("Synthèse des Risques", self.styles['SectionHeader']))
        story.append(Paragraph(risk_summary, self.styles['HighlightBox']))
        
        # Top risks
        top_risks = risk_analysis.get("top_risks", [])
        if top_risks and isinstance(top_risks, list):
            story.append(Paragraph("Risques Prioritaires", self.styles['SectionHeader']))
            for i, risk in enumerate(top_risks[:5], 1):  # Top 5 risks
                story.append(Paragraph(f"{i}. {risk}", self.styles['Normal']))
        
        # Mitigation strategies
        strategies = risk_analysis.get("mitigation_strategies", [])
        if strategies and isinstance(strategies, list):
            story.append(Spacer(1, 12))
            story.append(Paragraph("Stratégies de Mitigation", self.styles['SectionHeader']))
            for strategy in strategies[:3]:  # Top 3 strategies
                story.append(Paragraph(f"• {strategy}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_ai_insights(self, plan_data: Dict[str, Any]) -> List:
        """Build AI insights section from all 20 AI systems"""
        story = []
        
        story.append(Paragraph("Intelligence Artificielle - Insights Multi-Agents", self.styles['CustomSubtitle']))
        
        # Strategy Insights
        strategy_insights = plan_data.get("strategy_insights")
        if strategy_insights:
            story.append(Paragraph("Conseiller Stratégique", self.styles['SectionHeader']))
            
            market_pos = strategy_insights.get("market_positioning", "Non défini")
            competitive = strategy_insights.get("competitive_advantage", "Non défini")
            success_prob = strategy_insights.get("success_probability", 0)
            
            strategy_text = f"""
            Positionnement Marché: {market_pos}<br/>
            Avantage Concurrentiel: {competitive}<br/>
            Probabilité de Succès: {success_prob}%
            """
            story.append(Paragraph(strategy_text, self.styles['HighlightBox']))
        
        # Learning Insights
        learning_insights = plan_data.get("learning_insights")
        if learning_insights:
            story.append(Paragraph("Agent d'Apprentissage Adaptatif", self.styles['SectionHeader']))
            
            skill_gaps = learning_insights.get("skill_gaps", [])
            if skill_gaps:
                story.append(Paragraph("Lacunes de Compétences Identifiées:", self.styles['Normal']))
                for gap in skill_gaps[:3]:
                    story.append(Paragraph(f"• {gap}", self.styles['Normal']))
        
        # Stakeholder Intelligence
        stakeholder_insights = plan_data.get("stakeholder_insights")
        if stakeholder_insights:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Intelligence des Parties Prenantes", self.styles['SectionHeader']))
            
            complexity = stakeholder_insights.get("stakeholder_complexity", "Moyen")
            story.append(Paragraph(f"Niveau de Complexité: {complexity}", self.styles['Normal']))
        
        # Innovation Catalyst
        innovation_insights = plan_data.get("innovation_insights")
        if innovation_insights:
            story.append(Spacer(1, 12))
            story.append(Paragraph("Catalyseur d'Innovation", self.styles['SectionHeader']))
            
            opportunities = innovation_insights.get("innovation_opportunities", [])
            if opportunities and isinstance(opportunities, list):
                story.append(Paragraph("Opportunités d'Innovation:", self.styles['Normal']))
                for opp in opportunities[:3]:
                    story.append(Paragraph(f"• {opp}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_gantt_chart(self, plan_data: Dict[str, Any]) -> List:
        """Build Gantt chart visualization"""
        story = []
        if not HAS_MATPLOTLIB:
            return story
        
        try:
            plan = plan_data.get("plan", {})
            phases = plan.get("wbs", {}).get("phases", [])
            
            if not phases:
                return story
            
            story.append(Paragraph("Diagramme de Gantt", self.styles['CustomSubtitle']))
            
            # Create Gantt chart
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Prepare data
            y_pos = 0
            colors_list = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
            
            for i, phase in enumerate(phases[:8]):  # Limit to 8 phases
                phase_name = phase.get("name", f"Phase {i+1}")
                tasks = phase.get("tasks", [])
                
                for j, task in enumerate(tasks[:5]):  # Limit to 5 tasks per phase
                    task_name = task.get("name", f"Task {j+1}")
                    duration = task.get("duration", 5)
                    start = j * 5  # Simple offset for visualization
                    
                    ax.barh(y_pos, duration, left=start, height=0.8,
                           color=colors_list[i % len(colors_list)], alpha=0.7)
                    ax.text(start + duration/2, y_pos, task_name[:20], 
                           ha='center', va='center', fontsize=8)
                    y_pos += 1
            
            ax.set_xlabel('Jours')
            ax.set_title('Planning du Projet')
            ax.grid(True, alpha=0.3)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Add to PDF
            img = Image(temp_file.name, width=6*inch, height=3.5*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            
            # Clean up
            os.unlink(temp_file.name)
            
        except Exception as e:
            logger.warning(f"Could not create Gantt chart: {e}")
        
        return story
    
    def _build_budget_chart(self, plan_data: Dict[str, Any]) -> List:
        """Build budget breakdown chart"""
        story = []
        if not HAS_MATPLOTLIB:
            return story
        
        try:
            plan = plan_data.get("plan", {})
            phases = plan.get("wbs", {}).get("phases", [])
            
            if not phases:
                return story
            
            story.append(Paragraph("Répartition du Budget", self.styles['CustomSubtitle']))
            
            # Calculate phase costs
            phase_names = []
            phase_costs = []
            
            for phase in phases[:6]:  # Limit to 6 phases
                phase_name = phase.get("name", "Phase")
                tasks = phase.get("tasks", [])
                total_cost = sum(task.get("cost", 0) for task in tasks)
                
                if total_cost > 0:
                    phase_names.append(phase_name[:15])
                    phase_costs.append(total_cost)
            
            if phase_costs:
                # Create pie chart
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                
                # Pie chart
                colors_list = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
                ax1.pie(phase_costs, labels=phase_names, autopct='%1.1f%%',
                       colors=colors_list[:len(phase_costs)])
                ax1.set_title('Répartition par Phase')
                
                # Bar chart
                ax2.bar(phase_names, phase_costs, color=colors_list[:len(phase_costs)])
                ax2.set_xlabel('Phases')
                ax2.set_ylabel('Coût (€)')
                ax2.set_title('Budget par Phase')
                ax2.tick_params(axis='x', rotation=45)
                
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                plt.tight_layout()
                plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
                plt.close()
                
                # Add to PDF
                img = Image(temp_file.name, width=6*inch, height=2.5*inch)
                story.append(img)
                story.append(Spacer(1, 20))
                
                # Clean up
                os.unlink(temp_file.name)
            
        except Exception as e:
            logger.warning(f"Could not create budget chart: {e}")
        
        return story
    
    def _build_risk_matrix(self, plan_data: Dict[str, Any]) -> List:
        """Build risk matrix visualization"""
        story = []
        if not HAS_MATPLOTLIB:
            return story
        
        try:
            risk_analysis = plan_data.get("risk_analysis", {})
            
            if not risk_analysis:
                return story
            
            story.append(Paragraph("Matrice des Risques", self.styles['SectionHeader']))
            
            # Create risk matrix
            fig, ax = plt.subplots(figsize=(6, 4))
            
            # Define risk zones
            ax.add_patch(Rectangle((0, 0), 1, 1, color='green', alpha=0.3))
            ax.add_patch(Rectangle((1, 0), 1, 1, color='yellow', alpha=0.3))
            ax.add_patch(Rectangle((2, 0), 1, 1, color='orange', alpha=0.3))
            ax.add_patch(Rectangle((0, 1), 1, 1, color='yellow', alpha=0.3))
            ax.add_patch(Rectangle((1, 1), 1, 1, color='orange', alpha=0.3))
            ax.add_patch(Rectangle((2, 1), 1, 1, color='red', alpha=0.3))
            ax.add_patch(Rectangle((0, 2), 1, 1, color='orange', alpha=0.3))
            ax.add_patch(Rectangle((1, 2), 1, 1, color='red', alpha=0.3))
            ax.add_patch(Rectangle((2, 2), 1, 1, color='darkred', alpha=0.3))
            
            # Add labels
            ax.set_xlim(0, 3)
            ax.set_ylim(0, 3)
            ax.set_xlabel('Probabilité')
            ax.set_ylabel('Impact')
            ax.set_title('Matrice des Risques du Projet')
            ax.set_xticks([0.5, 1.5, 2.5])
            ax.set_xticklabels(['Faible', 'Moyen', 'Élevé'])
            ax.set_yticks([0.5, 1.5, 2.5])
            ax.set_yticklabels(['Faible', 'Moyen', 'Élevé'])
            
            # Add risk points (example)
            risk_score = risk_analysis.get("risk_score", 5)
            if risk_score < 4:
                ax.plot(0.5, 0.5, 'ko', markersize=10)
            elif risk_score < 7:
                ax.plot(1.5, 1.5, 'ko', markersize=10)
            else:
                ax.plot(2.5, 2.5, 'ko', markersize=10)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            plt.savefig(temp_file.name, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Add to PDF
            img = Image(temp_file.name, width=4*inch, height=2.5*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            
            # Clean up
            os.unlink(temp_file.name)
            
        except Exception as e:
            logger.warning(f"Could not create risk matrix: {e}")
        
        return story
    
    def _build_critical_path_analysis(self, plan_data: Dict[str, Any]) -> List:
        """Build critical path analysis section"""
        story = []
        plan = plan_data.get("plan", {})
        
        story.append(Paragraph("Analyse du Chemin Critique", self.styles['CustomSubtitle']))
        
        # Critical path information
        critical_path = plan.get("critical_path", {})
        if critical_path:
            story.append(Paragraph("Tâches sur le Chemin Critique", self.styles['SectionHeader']))
            
            critical_tasks = critical_path.get("tasks", [])
            if critical_tasks:
                for i, task in enumerate(critical_tasks[:10], 1):
                    task_info = f"{i}. {task.get('name', 'Tâche')} - Durée: {task.get('duration', 0)} jours"
                    story.append(Paragraph(task_info, self.styles['Normal']))
            
            story.append(Spacer(1, 12))
            
            # Float analysis
            story.append(Paragraph("Analyse des Marges", self.styles['SectionHeader']))
            total_float = critical_path.get("total_float", 0)
            story.append(Paragraph(f"Marge totale du projet: {total_float} jours", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_resource_allocation(self, plan_data: Dict[str, Any]) -> List:
        """Build resource allocation section"""
        story = []
        plan = plan_data.get("plan", {})
        
        story.append(Paragraph("Allocation des Ressources", self.styles['CustomSubtitle']))
        
        # Collect resource information
        resources = {}
        phases = plan.get("wbs", {}).get("phases", [])
        
        for phase in phases:
            tasks = phase.get("tasks", [])
            for task in tasks:
                assigned = task.get("assigned_resources", [])
                for resource in assigned:
                    if resource not in resources:
                        resources[resource] = {"tasks": 0, "hours": 0, "cost": 0}
                    resources[resource]["tasks"] += 1
                    resources[resource]["hours"] += task.get("duration", 0) * 8
                    resources[resource]["cost"] += task.get("cost", 0)
        
        if resources:
            # Create resource table
            resource_data = [['Ressource', 'Tâches', 'Heures', 'Coût (€)']]
            
            for resource, data in list(resources.items())[:10]:  # Limit to 10 resources
                resource_data.append([
                    resource[:30],
                    str(data["tasks"]),
                    f"{data['hours']:.0f}",
                    f"€{data['cost']:,.0f}"
                ])
            
            resource_table = Table(resource_data, colWidths=[2.5*inch, 1*inch, 1*inch, 1.5*inch])
            resource_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#5DADE2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F0F8FF')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#5DADE2'))
            ]))
            
            story.append(resource_table)
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_quality_metrics(self, plan_data: Dict[str, Any]) -> List:
        """Build quality metrics section"""
        story = []
        
        story.append(Paragraph("Métriques de Qualité", self.styles['CustomSubtitle']))
        
        # Quality indicators
        quality_metrics = plan_data.get("quality_metrics", {})
        
        metrics_text = f"""
        • Conformité aux Standards: {quality_metrics.get('standards_compliance', 95)}%
        • Couverture des Tests: {quality_metrics.get('test_coverage', 80)}%
        • Indice de Satisfaction: {quality_metrics.get('satisfaction_index', 4.5)}/5
        • Taux de Défauts Prévus: {quality_metrics.get('defect_rate', 2)}%
        • Score de Maturité: {quality_metrics.get('maturity_score', 3)}/5
        """
        
        story.append(Paragraph("Indicateurs de Qualité", self.styles['SectionHeader']))
        story.append(Paragraph(metrics_text, self.styles['HighlightBox']))
        
        story.append(Spacer(1, 20))
        return story
    
    def _build_appendix(self, plan_data: Dict[str, Any]) -> List:
        """Build appendix with additional data"""
        story = []
        
        story.append(PageBreak())
        story.append(Paragraph("Annexes", self.styles['CustomSubtitle']))
        
        # Glossary
        story.append(Paragraph("Glossaire", self.styles['SectionHeader']))
        glossary = [
            ("WBS", "Work Breakdown Structure - Structure de découpage du travail"),
            ("ROI", "Return On Investment - Retour sur investissement"),
            ("KPI", "Key Performance Indicator - Indicateur clé de performance"),
            ("SLA", "Service Level Agreement - Accord de niveau de service"),
            ("RAG", "Retrieval-Augmented Generation - Génération augmentée par récupération")
        ]
        
        for term, definition in glossary:
            story.append(Paragraph(f"<b>{term}</b>: {definition}", self.styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Metadata
        story.append(Paragraph("Métadonnées du Projet", self.styles['SectionHeader']))
        metadata = plan_data.get("metadata", {})
        
        metadata_text = f"""
        • Version du Plan: {metadata.get('version', '1.0')}
        • Date de Création: {metadata.get('created_at', datetime.now().strftime('%Y-%m-%d'))}
        • Dernière Modification: {metadata.get('updated_at', datetime.now().strftime('%Y-%m-%d'))}
        • Généré par: PlannerIA v2.0
        • Nombre d'Agents IA: 20
        """
        
        story.append(Paragraph(metadata_text, self.styles['Normal']))
        
        return story
    
    def _build_recommendations(self, plan_data: Dict[str, Any]) -> List:
        """Build recommendations section"""
        story = []
        
        story.append(Paragraph("Recommandations", self.styles['CustomSubtitle']))
        
        # Collect recommendations from various sources
        recommendations = []
        
        # Strategy recommendations
        strategy_insights = plan_data.get("strategy_insights", {})
        strategy_recs = strategy_insights.get("strategic_recommendations", [])
        if isinstance(strategy_recs, list):
            recommendations.extend(strategy_recs[:2])  # Top 2
        
        # Learning recommendations  
        learning_insights = plan_data.get("learning_insights", {})
        learning_recs = learning_insights.get("learning_recommendations", [])
        if isinstance(learning_recs, list):
            recommendations.extend(learning_recs[:2])  # Top 2
        
        # Monitor recommendations
        monitor_insights = plan_data.get("monitor_insights", {})
        monitor_recs = monitor_insights.get("recommendations", [])
        if isinstance(monitor_recs, list):
            recommendations.extend(monitor_recs[:2])  # Top 2
        
        # Display recommendations
        if recommendations:
            story.append(Paragraph("Recommandations Prioritaires", self.styles['SectionHeader']))
            for i, rec in enumerate(recommendations[:6], 1):  # Max 6 recommendations
                story.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
        else:
            story.append(Paragraph("Aucune recommandation spécifique générée.", self.styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph("--- Fin du Rapport ---", self.styles['Normal']))
        story.append(Paragraph("Généré par PlannerIA - Système Multi-Agents d'Intelligence Artificielle", 
                              self.styles['Normal']))
        
        return story


def generate_pdf_report(plan_data: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Convenience function to generate PDF report
    
    Args:
        plan_data: Complete plan data with all insights
        output_path: Optional output path
        
    Returns:
        Path to generated PDF file
    """
    generator = PDFReportGenerator()
    return generator.generate_report(plan_data, output_path)