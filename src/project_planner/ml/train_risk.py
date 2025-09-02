"""
Train Risk Models - Script to train ML models for risk prediction and classification
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.project_planner.ml.risk_model import RiskModel


def generate_synthetic_risk_training_data(num_samples: int = 300) -> List[Dict[str, Any]]:
    """Generate synthetic training data for risk models"""
    
    print(f"ðŸ”§ Generating {num_samples} synthetic project samples with risks...")
    
    # Project archetypes with typical risk profiles
    project_types = {
        'web_application': {
            'title_templates': [
                'E-commerce Platform', 'Customer Portal', 'Content Management System',
                'Social Media Platform', 'Business Dashboard', 'Online Marketplace'
            ],
            'common_risks': ['technical', 'schedule', 'quality'],
            'complexity_factor': 1.2,
            'typical_duration': 60,
            'typical_cost': 150000
        },
        'mobile_app': {
            'title_templates': [
                'Mobile Banking App', 'Fitness Tracking App', 'Food Delivery App',
                'Travel Booking App', 'Educational App', 'Gaming Application'
            ],
            'common_risks': ['technical', 'resource', 'quality'],
            'complexity_factor': 1.5,
            'typical_duration': 90,
            'typical_cost': 200000
        },
        'enterprise_system': {
            'title_templates': [
                'ERP Implementation', 'CRM Integration', 'Data Warehouse',
                'Legacy System Migration', 'Business Intelligence Platform'
            ],
            'common_risks': ['technical', 'schedule', 'budget', 'external'],
            'complexity_factor': 2.0,
            'typical_duration': 180,
            'typical_cost': 500000
        },
        'infrastructure': {
            'title_templates': [
                'Cloud Migration', 'DevOps Pipeline', 'Security Implementation',
                'Network Upgrade', 'Server Consolidation', 'Monitoring System'
            ],
            'common_risks': ['technical', 'schedule', 'external'],
            'complexity_factor': 1.8,
            'typical_duration': 120,
            'typical_cost': 300000
        },
        'data_science': {
            'title_templates': [
                'ML Model Development', 'Data Analytics Platform', 'AI Chatbot',
                'Predictive Analytics', 'Recommendation Engine', 'Data Pipeline'
            ],
            'common_risks': ['technical', 'resource', 'quality'],
            'complexity_factor': 1.7,
            'typical_duration': 100,
            'typical_cost': 250000
        }
    }
    
    # Risk templates for each category
    risk_templates = {
        'technical': [
            {
                'name': 'Technology Integration Complexity',
                'description': 'Complex integration between multiple systems may cause delays',
                'base_probability': 3,
                'base_impact': 3
            },
            {
                'name': 'Performance Requirements',
                'description': 'System may not meet performance benchmarks under load',
                'base_probability': 2,
                'base_impact': 4
            },
            {
                'name': 'Third-party Dependencies',
                'description': 'External APIs and services may have reliability issues',
                'base_probability': 3,
                'base_impact': 3
            }
        ],
        'schedule': [
            {
                'name': 'Aggressive Timeline',
                'description': 'Tight deadlines may not allow sufficient development time',
                'base_probability': 4,
                'base_impact': 4
            },
            {
                'name': 'Dependency Delays',
                'description': 'External dependencies may cause project timeline delays',
                'base_probability': 3,
                'base_impact': 3
            },
            {
                'name': 'Resource Scheduling Conflicts',
                'description': 'Team members may have competing priorities',
                'base_probability': 3,
                'base_impact': 3
            }
        ],
        'budget': [
            {
                'name': 'Cost Overrun Risk',
                'description': 'Project costs may exceed initial budget estimates',
                'base_probability': 3,
                'base_impact': 4
            },
            {
                'name': 'Scope Creep',
                'description': 'Additional requirements may increase project cost',
                'base_probability': 4,
                'base_impact': 3
            },
            {
                'name': 'Vendor Pricing Changes',
                'description': 'Third-party service costs may increase unexpectedly',
                'base_probability': 2,
                'base_impact': 3
            }
        ],
        'resource': [
            {
                'name': 'Key Personnel Availability',
                'description': 'Critical team members may become unavailable',
                'base_probability': 3,
                'base_impact': 4
            },
            {
                'name': 'Skill Gap Issues',
                'description': 'Team may lack specific technical expertise required',
                'base_probability': 3,
                'base_impact': 3
            },
            {
                'name': 'Team Capacity Constraints',
                'description': 'Development team may be over-allocated',
                'base_probability': 4,
                'base_impact': 3
            }
        ],
        'external': [
            {
                'name': 'Vendor Reliability Issues',
                'description': 'External vendors may not deliver on schedule',
                'base_probability': 2,
                'base_impact': 4
            },
            {
                'name': 'Regulatory Changes',
                'description': 'New compliance requirements may emerge',
                'base_probability': 2,
                'base_impact': 4
            },
            {
                'name': 'Market Condition Changes',
                'description': 'Business environment changes may affect requirements',
                'base_probability': 2,
                'base_impact': 3
            }
        ],
        'quality': [
            {
                'name': 'User Acceptance Issues',
                'description': 'End users may not accept the delivered solution',
                'base_probability': 3,
                'base_impact': 4
            },
            {
                'name': 'Quality Standard Compliance',
                'description': 'Solution may not meet quality requirements',
                'base_probability': 2,
                'base_impact': 4
            },
            {
                'name': 'Testing Coverage Gaps',
                'description': 'Inadequate testing may lead to production issues',
                'base_probability': 3,
                'base_impact': 3
            }
        ]
    }
    
    training_data = []
    
    for i in range(num_samples):
        # Select project type
        project_type = random.choice(list(project_types.keys()))
        project_info = project_types[project_type]
        
        # Generate project overview
        title = random.choice(project_info['title_templates'])
        
        # Add complexity variation
        complexity_multiplier = random.uniform(0.7, 1.5)
        duration = int(project_info['typical_duration'] * complexity_multiplier)
        cost = int(project_info['typical_cost'] * complexity_multiplier)
        
        # Generate tasks
        num_tasks = random.randint(8, 25)
        tasks = []
        
        for j in range(num_tasks):
            task_duration = random.uniform(1, 8)
            task_cost = random.uniform(800, 5000)
            
            task = {
                'id': f'task_{j:02d}',
                'name': f'Task {j+1}',
                'description': f'Development task for {title}',
                'duration': task_duration,
                'cost': task_cost,
                'priority': random.choice(['low', 'medium', 'high', 'critical'])
            }
            tasks.append(task)
        
        # Generate dependencies
        num_dependencies = random.randint(0, min(8, num_tasks - 1))
        dependencies = []
        
        for k in range(num_dependencies):
            pred_idx = random.randint(0, num_tasks - 2)
            succ_idx = random.randint(pred_idx + 1, num_tasks - 1)
            
            dependency = {
                'predecessor': f'task_{pred_idx:02d}',
                'successor': f'task_{succ_idx:02d}',
                'type': 'finish_to_start'
            }
            dependencies.append(dependency)
        
        # Generate risks based on project type
        risks = []
        num_risks = random.randint(2, 7)
        
        # Select risk categories (favor common risks for project type)
        available_categories = list(risk_templates.keys())
        common_risks = project_info['common_risks']
        
        selected_categories = []
        for _ in range(num_risks):
            if len(selected_categories) < len(common_risks) and random.random() > 0.3:
                # Favor common risks
                category = random.choice([c for c in common_risks if c not in selected_categories])
            else:
                # Add other risk types
                category = random.choice(available_categories)
            
            selected_categories.append(category)
        
        # Generate risk instances
        for k, category in enumerate(selected_categories):
            risk_template = random.choice(risk_templates[category])
            
            # Add variation to probability and impact
            prob_variation = random.randint(-1, 1)
            impact_variation = random.randint(-1, 1)
            
            probability = max(1, min(5, risk_template['base_probability'] + prob_variation))
            impact = max(1, min(5, risk_template['base_impact'] + impact_variation))
            
            # Project complexity affects risk levels
            if project_info['complexity_factor'] > 1.5:
                probability = min(5, probability + 1)
            
            risk = {
                'id': f'risk_{k:03d}',
                'name': risk_template['name'],
                'description': risk_template['description'],
                'category': category,
                'probability': probability,
                'impact': impact,
                'risk_score': probability * impact,
                'mitigation_strategy': f'Implement {category} risk mitigation measures',
                'contingency_plan': f'Activate {category} contingency procedures',
                'status': 'identified'
            }
            risks.append(risk)
        
        # Generate resources
        num_resources = random.randint(3, 8)
        resources = []
        
        for r in range(num_resources):
            resource = {
                'id': f'resource_{r}',
                'name': f'Team Member {r+1}',
                'type': 'human',
                'cost_per_unit': random.uniform(80, 150),
                'availability': random.uniform(0.7, 1.0)
            }
            resources.append(resource)
        
        # Create project
        project = {
            'run_id': f'synthetic_project_{i:04d}',
            'timestamp': datetime.now().isoformat(),
            'brief': f'Develop {title} with {project_type} requirements',
            'project_overview': {
                'title': title,
                'description': f'A comprehensive {project_type} project requiring {duration} days',
                'total_duration': duration,
                'total_cost': cost
            },
            'wbs': {
                'phases': [{
                    'id': 'main_phase',
                    'name': 'Development Phase',
                    'tasks': tasks
                }]
            },
            'tasks': tasks,
            'dependencies': dependencies,
            'risks': risks,
            'resources': resources,
            'project_type': project_type,
            'complexity_factor': project_info['complexity_factor']
        }
        
        training_data.append(project)
    
    print(f"âœ… Generated {len(training_data)} synthetic project samples")
    
    # Display risk distribution
    risk_categories = {}
    for project in training_data:
        for risk in project['risks']:
            cat = risk['category']
            risk_categories[cat] = risk_categories.get(cat, 0) + 1
    
    print("ðŸ“Š Risk category distribution:")
    for category, count in sorted(risk_categories.items()):
        print(f"   â€¢ {category}: {count} risks")
    
    return training_data


def load_historical_project_data() -> List[Dict[str, Any]]:
    """Load historical project data if available"""
    
    training_dir = Path("data/training")
    historical_projects = []
    
    if training_dir.exists():
        for data_file in training_dir.glob("*risk*.json"):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        historical_projects.extend(data)
                    elif isinstance(data, dict):
                        historical_projects.append(data)
                        
                print(f"ðŸ“ Loaded historical data from {data_file.name}")
                
            except Exception as e:
                print(f"âš ï¸ Error loading {data_file}: {e}")
        
        # Also check for existing project plans
        runs_dir = Path("data/runs")
        if runs_dir.exists():
            for run_dir in runs_dir.iterdir():
                if run_dir.is_dir():
                    plan_file = run_dir / "plan.json"
                    if plan_file.exists():
                        try:
                            with open(plan_file, 'r', encoding='utf-8') as f:
                                project_data = json.load(f)
                                if project_data.get('risks'):  # Only use if has risks
                                    historical_projects.append(project_data)
                        except Exception:
                            continue
    
    if historical_projects:
        print(f"ðŸ“Š Found {len(historical_projects)} historical projects")
    
    return historical_projects


def save_training_data(data: List[Dict[str, Any]], filename: str = "risk_training_data.json"):
    """Save training data for future reference"""
    
    training_dir = Path("data/training")
    training_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = training_dir / filename
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"ðŸ’¾ Saved risk training data to {output_file}")


def evaluate_risk_model(model: RiskModel, test_projects: List[Dict[str, Any]]):
    """Evaluate risk model performance"""
    
    print("ðŸ“Š Evaluating risk model performance...")
    
    total_predictions = 0
    correct_categories = 0
    risk_coverage = 0
    
    for project in test_projects:
        actual_risks = project.get('risks', [])
        actual_categories = set(risk.get('category') for risk in actual_risks)
        
        # Predict risks
        predicted_risks = model.predict_project_risks(project)
        predicted_categories = set(risk.get('category') for risk in predicted_risks)
        
        total_predictions += 1
        
        # Check category overlap
        if actual_categories & predicted_categories:  # Any intersection
            correct_categories += 1
        
        # Check coverage
        if len(predicted_categories & actual_categories) > 0:
            coverage = len(predicted_categories & actual_categories) / len(actual_categories)
            risk_coverage += coverage
    
    if total_predictions > 0:
        category_accuracy = correct_categories / total_predictions * 100
        avg_coverage = risk_coverage / total_predictions * 100
        
        print(f"ðŸ“ˆ Risk Category Detection: {category_accuracy:.1f}%")
        print(f"ðŸŽ¯ Risk Coverage: {avg_coverage:.1f}%")
        print(f"ðŸ“ Evaluated on {total_predictions} test projects")
    
    # Show sample predictions
    print("\nðŸ” Sample Risk Predictions:")
    sample_project = test_projects[0] if test_projects else None
    
    if sample_project:
        predicted_risks = model.predict_project_risks(sample_project)
        print(f"Project: {sample_project.get('project_overview', {}).get('title', 'Unknown')}")
        
        for risk in predicted_risks[:3]:  # Show top 3
            print(f"   â€¢ {risk['name']} (Score: {risk['risk_score']}, Confidence: {risk['confidence']:.2f})")


def main():
    """Main training script for risk models"""
    
    print("ðŸš€ Starting Risk Model Training")
    print("=" * 50)
    
    try:
        # Initialize risk model
        model = RiskModel()
        
        # Load or generate training data
        historical_data = load_historical_project_data()
        
        if len(historical_data) < 50:
            print(f"ðŸ“ Insufficient historical data ({len(historical_data)} projects)")
            print("ðŸ”§ Generating synthetic training data...")
            synthetic_data = generate_synthetic_risk_training_data(300)
            
            # Combine with historical data
            training_data = historical_data + synthetic_data
        else:
            print(f"ðŸ“Š Using {len(historical_data)} historical projects")
            training_data = historical_data
        
        # Save training data
        save_training_data(training_data)
        
        # Split for evaluation
        np.random.seed(42)
        np.random.shuffle(training_data)
        
        train_size = int(0.8 * len(training_data))
        train_projects = training_data[:train_size]
        test_projects = training_data[train_size:]
        
        print(f"ðŸ“š Training projects: {len(train_projects)}")
        print(f"ðŸ§ª Test projects: {len(test_projects)}")
        
        # Train the model
        print("\nðŸ‹ï¸ Training risk models...")
        model_info = model.train(train_projects)
        
        print(f"âœ… Training completed!")
        print(f"ðŸ“… Training date: {model_info['training_date']}")
        print(f"ðŸŽ¯ Category accuracy: {model_info['performance_metrics']['category_accuracy']:.3f}")
        print(f"ðŸ“Š Severity accuracy: {model_info['performance_metrics']['severity_accuracy']:.3f}")
        
        # Evaluate on test data
        if test_projects:
            print("\nðŸ“Š Evaluating on test data...")
            evaluate_risk_model(model, test_projects)
        
        # Risk pattern analysis
        print("\nðŸ” Analyzing risk patterns...")
        risk_analysis = model.analyze_risk_patterns(training_data)
        
        print("ðŸ“ˆ Risk frequency by category:")
        for category, freq in sorted(risk_analysis['risk_frequency'].items()):
            avg_score = risk_analysis['avg_risk_scores'].get(category, 0)
            print(f"   â€¢ {category}: {freq} occurrences (avg score: {avg_score:.1f})")
        
        print("\nðŸŽ‰ Risk model training completed successfully!")
        print(f"ðŸ“ Models saved to: {model.model_path}")
        
        # Usage example
        print("\nðŸ’¡ Usage Example:")
        print("```python")
        print("from src.project_planner.ml.risk_model import RiskModel")
        print("model = RiskModel()")
        print("risks = model.predict_project_risks({")
        print("    'project_overview': {'title': 'Web Application', 'total_cost': 100000},")
        print("    'tasks': [{'name': 'Frontend Development', 'duration': 30}]")
        print("})")
        print("for risk in risks:")
        print("    print(f'{risk[\"name\"]}: {risk[\"risk_score\"]} (confidence: {risk[\"confidence\"]:.2f})')")
        print("```")
        
    except Exception as e:
        print(f"âŒ Risk model training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

