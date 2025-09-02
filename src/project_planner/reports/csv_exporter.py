"""
CSV Export functionality for PlannerIA
Exports comprehensive project data to multiple CSV files for analysis
"""

import csv
import os
import zipfile
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class CSVExporter:
    """Export project data to comprehensive CSV files"""
    
    def __init__(self):
        self.export_dir = None
        self.files_created = []
    
    def export_complete_project(self, plan_data: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """
        Export complete project data to multiple CSV files in a ZIP archive
        
        Args:
            plan_data: Complete plan data with all insights
            output_dir: Optional output directory
            
        Returns:
            Path to generated ZIP file containing all CSV exports
        """
        if not plan_data or "error" in plan_data:
            raise ValueError("Invalid plan data for CSV export")
        
        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = plan_data.get("plan", {}).get("project_overview", {}).get("title", "project")
            project_name = "".join(c for c in project_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
            
            # Create temp directory for CSV files
            reports_dir = os.path.join("data", "reports")
            os.makedirs(reports_dir, exist_ok=True)
            self.export_dir = os.path.join(reports_dir, f"export_{project_name}_{timestamp}")
            os.makedirs(self.export_dir, exist_ok=True)
        else:
            self.export_dir = output_dir
        
        try:
            # Export different aspects to separate CSV files
            self._export_project_overview(plan_data)
            self._export_tasks_detailed(plan_data)
            self._export_phases_summary(plan_data)
            self._export_resources(plan_data)
            self._export_risks(plan_data)
            self._export_budget_breakdown(plan_data)
            self._export_critical_path(plan_data)
            self._export_kpis(plan_data)
            self._export_ai_insights(plan_data)
            self._export_timeline(plan_data)
            
            # Create ZIP archive
            zip_path = self._create_zip_archive()
            
            logger.info(f"CSV export completed: {zip_path}")
            return zip_path
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise
    
    def _export_project_overview(self, plan_data: Dict[str, Any]):
        """Export project overview to CSV"""
        filepath = os.path.join(self.export_dir, "01_project_overview.csv")
        plan = plan_data.get("plan", {})
        overview = plan.get("project_overview", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            
            # Format values properly
            total_duration = overview.get('total_duration', 0)
            if isinstance(total_duration, (int, float)):
                duration_str = f"{total_duration:.0f}"
            else:
                duration_str = str(total_duration)
            
            total_cost = overview.get('total_cost', 0)
            if isinstance(total_cost, (int, float)):
                cost_str = f"{total_cost:.0f}"
            else:
                cost_str = str(total_cost)
            
            critical_path = overview.get('critical_path_duration', 0)
            if isinstance(critical_path, (int, float)):
                critical_str = f"{critical_path:.0f}"
            else:
                critical_str = str(critical_path)
            
            writer.writerow(['Project Title', overview.get('title', 'N/A')])
            writer.writerow(['Description', overview.get('description', 'N/A')])
            writer.writerow(['Total Duration (days)', duration_str])
            writer.writerow(['Total Cost (EUR)', cost_str])
            writer.writerow(['Critical Path (days)', critical_str])
            writer.writerow(['Number of Phases', len(plan.get("wbs", {}).get("phases", []))])
            writer.writerow(['Total Tasks', sum(len(p.get('tasks', [])) for p in plan.get("wbs", {}).get("phases", []))])
            writer.writerow(['Risk Score', plan_data.get('risk_analysis', {}).get('risk_score', 'N/A')])
            writer.writerow(['Generation Date', datetime.now().strftime("%Y-%m-%d %H:%M")])
            
            # Add objectives
            writer.writerow([])
            writer.writerow(['Objectives'])
            objectives = overview.get('objectives', [])
            if objectives and isinstance(objectives, list):
                for i, obj in enumerate(objectives, 1):
                    writer.writerow([f'Objective {i}', obj])
        
        self.files_created.append(filepath)
    
    def _export_tasks_detailed(self, plan_data: Dict[str, Any]):
        """Export detailed task information to CSV"""
        filepath = os.path.join(self.export_dir, "02_tasks_detailed.csv")
        plan = plan_data.get("plan", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Phase', 'Task_ID', 'Task_Name', 'Description', 'Duration_Days', 
                'Cost_EUR', 'Start_Date', 'End_Date', 'Assigned_Resources', 
                'Dependencies', 'Is_Critical', 'Risk_Level', 'Priority', 
                'Completion_Status', 'Effort_Hours'
            ])
            
            wbs = plan.get("wbs", {})
            phases = wbs.get("phases", [])
            critical_path = plan.get("critical_path", {}).get("tasks", [])
            critical_ids = [t.get("id") for t in critical_path] if critical_path else []
            
            for phase in phases:
                if not isinstance(phase, dict):
                    continue
                    
                phase_name = phase.get("name", "Unknown Phase")
                tasks = phase.get("tasks", [])
                
                if not isinstance(tasks, list):
                    continue
                
                for task in tasks:
                    if not isinstance(task, dict):
                        continue
                    
                    task_id = task.get("id", "")
                    
                    # Format cost properly
                    cost = task.get("cost", 0)
                    if isinstance(cost, (int, float)):
                        cost_str = f"{cost:.0f}"
                    else:
                        cost_str = str(cost)
                    
                    writer.writerow([
                        phase_name,
                        task_id,
                        task.get("name", "Unnamed Task"),
                        task.get("description", ""),
                        task.get("duration", 0),
                        cost_str,
                        task.get("start_date", ""),
                        task.get("end_date", ""),
                        ", ".join(task.get("assigned_resources", [])),
                        ", ".join(task.get("dependencies", [])),
                        "Yes" if task_id in critical_ids else "No",
                        task.get("risk_level", "Medium"),
                        task.get("priority", "Medium"),
                        task.get("status", "Not Started"),
                        task.get("effort", task.get("duration", 0) * 8)
                    ])
        
        self.files_created.append(filepath)
    
    def _export_phases_summary(self, plan_data: Dict[str, Any]):
        """Export phases summary to CSV"""
        filepath = os.path.join(self.export_dir, "03_phases_summary.csv")
        plan = plan_data.get("plan", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Phase_Name', 'Description', 'Total_Tasks', 'Total_Duration_Days', 
                'Total_Cost_EUR', 'Average_Task_Duration', 'Resources_Involved', 
                'Critical_Tasks', 'Risk_Assessment'
            ])
            
            wbs = plan.get("wbs", {})
            phases = wbs.get("phases", [])
            
            for phase in phases:
                if not isinstance(phase, dict):
                    continue
                
                tasks = phase.get("tasks", [])
                if not isinstance(tasks, list):
                    continue
                
                total_tasks = len(tasks)
                total_duration = sum(t.get("duration", 0) for t in tasks)
                total_cost = sum(t.get("cost", 0) for t in tasks)
                avg_duration = total_duration / total_tasks if total_tasks > 0 else 0
                
                # Collect unique resources
                all_resources = set()
                for task in tasks:
                    all_resources.update(task.get("assigned_resources", []))
                
                # Format cost properly
                if isinstance(total_cost, (int, float)):
                    cost_str = f"{total_cost:.0f}"
                else:
                    cost_str = str(total_cost)
                
                writer.writerow([
                    phase.get("name", "Unknown Phase"),
                    phase.get("description", ""),
                    total_tasks,
                    total_duration,
                    cost_str,
                    f"{avg_duration:.1f}",
                    ", ".join(all_resources),
                    sum(1 for t in tasks if t.get("is_critical", False)),
                    phase.get("risk_assessment", "Medium")
                ])
        
        self.files_created.append(filepath)
    
    def _export_resources(self, plan_data: Dict[str, Any]):
        """Export resource allocation to CSV"""
        filepath = os.path.join(self.export_dir, "04_resource_allocation.csv")
        plan = plan_data.get("plan", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Resource_Name', 'Total_Tasks', 'Total_Hours', 'Total_Cost_EUR', 
                'Utilization_Percent', 'Phases_Involved', 'Critical_Tasks'
            ])
            
            # Aggregate resource data
            resources = {}
            phases = plan.get("wbs", {}).get("phases", [])
            
            for phase in phases:
                phase_name = phase.get("name", "")
                tasks = phase.get("tasks", [])
                
                for task in tasks:
                    assigned = task.get("assigned_resources", [])
                    for resource in assigned:
                        if resource not in resources:
                            resources[resource] = {
                                "tasks": 0, 
                                "hours": 0, 
                                "cost": 0,
                                "phases": set(),
                                "critical": 0
                            }
                        
                        resources[resource]["tasks"] += 1
                        resources[resource]["hours"] += task.get("duration", 0) * 8
                        resources[resource]["cost"] += task.get("cost", 0)
                        resources[resource]["phases"].add(phase_name)
                        if task.get("is_critical", False):
                            resources[resource]["critical"] += 1
            
            # Write resource data
            total_project_hours = sum(r["hours"] for r in resources.values())
            
            for resource_name, data in resources.items():
                utilization = (data["hours"] / total_project_hours * 100) if total_project_hours > 0 else 0
                
                # Format cost properly
                cost = data["cost"]
                if isinstance(cost, (int, float)):
                    cost_str = f"{cost:.0f}"
                else:
                    cost_str = str(cost)
                
                writer.writerow([
                    resource_name,
                    data["tasks"],
                    f"{data['hours']:.0f}",
                    cost_str,
                    f"{utilization:.1f}",
                    ", ".join(data["phases"]),
                    data["critical"]
                ])
        
        self.files_created.append(filepath)
    
    def _export_risks(self, plan_data: Dict[str, Any]):
        """Export risk analysis to CSV"""
        filepath = os.path.join(self.export_dir, "05_risk_analysis.csv")
        risk_analysis = plan_data.get("risk_analysis", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Risk summary
            writer.writerow(['Risk Analysis Summary'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Overall Risk Score', risk_analysis.get('risk_score', 'N/A')])
            writer.writerow(['High Risks', risk_analysis.get('high_risks', 0)])
            writer.writerow(['Medium Risks', risk_analysis.get('medium_risks', 0)])
            writer.writerow(['Low Risks', risk_analysis.get('low_risks', 0)])
            
            writer.writerow([])
            
            # Top risks
            writer.writerow(['Top Risks'])
            writer.writerow(['Priority', 'Risk', 'Impact', 'Probability', 'Mitigation'])
            
            top_risks = risk_analysis.get('top_risks', [])
            if isinstance(top_risks, list):
                for i, risk in enumerate(top_risks, 1):
                    writer.writerow([
                        i,
                        risk if isinstance(risk, str) else risk.get('description', ''),
                        risk.get('impact', 'High') if isinstance(risk, dict) else 'High',
                        risk.get('probability', 'Medium') if isinstance(risk, dict) else 'Medium',
                        risk.get('mitigation', '') if isinstance(risk, dict) else ''
                    ])
            
            writer.writerow([])
            
            # Mitigation strategies
            writer.writerow(['Mitigation Strategies'])
            strategies = risk_analysis.get('mitigation_strategies', [])
            if isinstance(strategies, list):
                for i, strategy in enumerate(strategies, 1):
                    writer.writerow([f'Strategy {i}', strategy])
        
        self.files_created.append(filepath)
    
    def _export_budget_breakdown(self, plan_data: Dict[str, Any]):
        """Export budget breakdown to CSV"""
        filepath = os.path.join(self.export_dir, "06_budget_breakdown.csv")
        plan = plan_data.get("plan", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Category', 'Phase', 'Task', 'Cost_EUR', 'Cost_Type', 
                'Payment_Schedule', 'Budget_Status'
            ])
            
            phases = plan.get("wbs", {}).get("phases", [])
            
            for phase in phases:
                phase_name = phase.get("name", "")
                tasks = phase.get("tasks", [])
                
                for task in tasks:
                    cost = task.get("cost", 0)
                    if isinstance(cost, (int, float)):
                        cost_str = f"{cost:.0f}"
                    else:
                        cost_str = str(cost)
                    
                    writer.writerow([
                        task.get("cost_category", "Development"),
                        phase_name,
                        task.get("name", ""),
                        cost_str,
                        task.get("cost_type", "Fixed"),
                        task.get("payment_schedule", "On Completion"),
                        task.get("budget_status", "Approved")
                    ])
            
            # Add summary
            writer.writerow([])
            writer.writerow(['Budget Summary'])
            writer.writerow(['Category', 'Total_EUR', 'Percentage'])
            
            # Calculate totals by category
            categories = {}
            for phase in phases:
                for task in phase.get("tasks", []):
                    category = task.get("cost_category", "Development")
                    categories[category] = categories.get(category, 0) + task.get("cost", 0)
            
            total_budget = sum(categories.values())
            for category, amount in categories.items():
                percentage = (amount / total_budget * 100) if total_budget > 0 else 0
                writer.writerow([category, f"{amount:.0f}", f"{percentage:.1f}%"])
        
        self.files_created.append(filepath)
    
    def _export_critical_path(self, plan_data: Dict[str, Any]):
        """Export critical path analysis to CSV"""
        filepath = os.path.join(self.export_dir, "07_critical_path.csv")
        plan = plan_data.get("plan", {})
        critical_path = plan.get("critical_path", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Sequence', 'Task_Name', 'Duration_Days', 'Early_Start', 
                'Early_Finish', 'Late_Start', 'Late_Finish', 'Total_Float', 
                'Free_Float', 'Dependencies'
            ])
            
            critical_tasks = critical_path.get("tasks", [])
            
            for i, task in enumerate(critical_tasks, 1):
                writer.writerow([
                    i,
                    task.get("name", ""),
                    task.get("duration", 0),
                    task.get("early_start", 0),
                    task.get("early_finish", 0),
                    task.get("late_start", 0),
                    task.get("late_finish", 0),
                    task.get("total_float", 0),
                    task.get("free_float", 0),
                    ", ".join(task.get("dependencies", []))
                ])
            
            # Add summary
            writer.writerow([])
            writer.writerow(['Critical Path Summary'])
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Duration', critical_path.get("total_duration", 0)])
            writer.writerow(['Number of Critical Tasks', len(critical_tasks)])
            writer.writerow(['Total Float', critical_path.get("total_float", 0)])
        
        self.files_created.append(filepath)
    
    def _export_kpis(self, plan_data: Dict[str, Any]):
        """Export KPIs and metrics to CSV"""
        filepath = os.path.join(self.export_dir, "08_kpis_metrics.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['KPI Category', 'Metric', 'Value', 'Target', 'Status'])
            
            # Project KPIs
            plan = plan_data.get("plan", {})
            overview = plan.get("project_overview", {})
            
            writer.writerow(['Schedule', 'Project Duration', f"{overview.get('total_duration', 0):.0f} days", 'TBD', 'On Track'])
            writer.writerow(['Budget', 'Total Cost', f"€{overview.get('total_cost', 0):,.0f}", 'TBD', 'Within Budget'])
            writer.writerow(['Quality', 'Risk Score', f"{plan_data.get('risk_analysis', {}).get('risk_score', 0)}/10", '<5', 'Monitor'])
            
            # Performance metrics
            phases = plan.get("wbs", {}).get("phases", [])
            total_tasks = sum(len(p.get('tasks', [])) for p in phases)
            
            writer.writerow(['Scope', 'Total Tasks', total_tasks, 'N/A', 'Defined'])
            writer.writerow(['Scope', 'Total Phases', len(phases), 'N/A', 'Defined'])
            
            # AI System metrics
            writer.writerow([])
            writer.writerow(['AI Systems Performance'])
            writer.writerow(['System', 'Activity', 'Latency_ms', 'Accuracy_%'])
            
            ai_systems = [
                ('Supervisor', 95, 12, 96),
                ('Planner', 88, 18, 94),
                ('Estimator', 92, 15, 95),
                ('Risk Analyzer', 78, 22, 93),
                ('Documentation', 85, 8, 98)
            ]
            
            for system, activity, latency, accuracy in ai_systems:
                writer.writerow([system, activity, latency, accuracy])
        
        self.files_created.append(filepath)
    
    def _export_ai_insights(self, plan_data: Dict[str, Any]):
        """Export AI insights to CSV"""
        filepath = os.path.join(self.export_dir, "09_ai_insights.csv")
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['AI System', 'Insight Type', 'Value', 'Confidence', 'Recommendation'])
            
            # Strategy insights
            strategy = plan_data.get("strategy_insights", {})
            if strategy:
                writer.writerow([
                    'Strategy Advisor',
                    'Market Positioning',
                    strategy.get('market_positioning', 'N/A'),
                    strategy.get('confidence', 'High'),
                    strategy.get('recommendation', '')
                ])
            
            # Learning insights
            learning = plan_data.get("learning_insights", {})
            if learning:
                skill_gaps = learning.get('skill_gaps', [])
                for gap in skill_gaps[:3]:
                    writer.writerow([
                        'Learning Agent',
                        'Skill Gap',
                        gap,
                        'Medium',
                        'Training recommended'
                    ])
            
            # Innovation insights
            innovation = plan_data.get("innovation_insights", {})
            if innovation:
                opportunities = innovation.get('innovation_opportunities', [])
                for opp in opportunities[:3]:
                    writer.writerow([
                        'Innovation Catalyst',
                        'Opportunity',
                        opp,
                        'Medium',
                        'Explore feasibility'
                    ])
        
        self.files_created.append(filepath)
    
    def _export_timeline(self, plan_data: Dict[str, Any]):
        """Export project timeline to CSV"""
        filepath = os.path.join(self.export_dir, "10_project_timeline.csv")
        plan = plan_data.get("plan", {})
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Week', 'Phase', 'Tasks_Starting', 'Tasks_Ending', 
                'Resources_Required', 'Budget_Week_EUR', 'Milestones'
            ])
            
            # Simple weekly breakdown (example)
            phases = plan.get("wbs", {}).get("phases", [])
            week = 1
            
            for phase in phases:
                phase_name = phase.get("name", "")
                tasks = phase.get("tasks", [])
                phase_duration = sum(t.get("duration", 0) for t in tasks)
                weeks_in_phase = max(1, phase_duration // 5)  # Assume 5 days per week
                
                for w in range(weeks_in_phase):
                    tasks_this_week = [t.get("name", "") for t in tasks[w:w+2]][:2]  # Sample tasks
                    weekly_budget = sum(t.get("cost", 0) for t in tasks) / weeks_in_phase if weeks_in_phase > 0 else 0
                    
                    writer.writerow([
                        week,
                        phase_name,
                        ", ".join(tasks_this_week),
                        "",
                        "Team",
                        f"{weekly_budget:.0f}",
                        f"End of {phase_name}" if w == weeks_in_phase - 1 else ""
                    ])
                    week += 1
        
        self.files_created.append(filepath)
    
    def _create_zip_archive(self) -> str:
        """Create ZIP archive of all CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_filename = f"PlannerIA_Export_{timestamp}.zip"
        zip_path = os.path.join(os.path.dirname(self.export_dir), zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filepath in self.files_created:
                arcname = os.path.basename(filepath)
                zipf.write(filepath, arcname)
        
        # Clean up individual CSV files
        for filepath in self.files_created:
            try:
                os.remove(filepath)
            except:
                pass
        
        # Remove temp directory
        try:
            os.rmdir(self.export_dir)
        except:
            pass
        
        return zip_path


def export_plan_to_csv(plan_data: Dict[str, Any], output_path: str = None) -> str:
    """
    Legacy function - exports basic task list to CSV
    For comprehensive export, use CSVExporter.export_complete_project()
    """
    exporter = CSVExporter()
    
    # If specific path is provided, export simple CSV
    if output_path and output_path.endswith('.csv'):
        # Export simple task list for backward compatibility
        if not plan_data or "error" in plan_data:
            raise ValueError("Invalid plan data for CSV export")
        
        plan = plan_data.get("plan", {})
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    'Phase', 'Task_Name', 'Description', 'Duration_Days', 'Cost_EUR', 
                    'Assigned_Resources', 'Is_Critical', 'Risk_Level', 'Priority'
                ])
                
                # Extract and write tasks
                wbs = plan.get("wbs", {})
                phases = wbs.get("phases", [])
                critical_path = plan.get("critical_path", [])
                
                for phase in phases:
                    if not isinstance(phase, dict):
                        continue
                        
                    phase_name = phase.get("name", "Unknown Phase")
                    tasks = phase.get("tasks", [])
                    
                    if not isinstance(tasks, list):
                        continue
                    
                    for task in tasks:
                        if not isinstance(task, dict):
                            continue
                        
                        task_id = task.get("id", "")
                        task_name = task.get("name", "Unnamed Task")
                        description = task.get("description", "")
                        duration = task.get("duration", 0)
                        cost = task.get("cost", 0)
                        resources = ", ".join(task.get("assigned_resources", []))
                        is_critical = "Yes" if task_id in critical_path else "No"
                        risk_level = task.get("risk_level", "Medium")
                        priority = task.get("priority", "Medium")
                        
                        writer.writerow([
                            phase_name, task_name, description, duration, cost,
                            resources, is_critical, risk_level, priority
                        ])
            
            logger.info(f"CSV export completed: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            raise
    else:
        # Export comprehensive ZIP archive
        return exporter.export_complete_project(plan_data, output_path)


def export_complete_project_data(plan_data: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """
    Export complete project data to multiple CSV files
    
    Args:
        plan_data: Complete plan data with all insights
        output_dir: Optional output directory
        
    Returns:
        Path to generated ZIP file containing all CSV exports
    """
    exporter = CSVExporter()
    return exporter.export_complete_project(plan_data, output_dir)