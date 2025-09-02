"""
ML Risk Assessment Model for PlannerIA
Provides machine learning-based risk classification and impact prediction.
"""

import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, mean_squared_error, accuracy_score
    from sklearn.multioutput import MultiOutputRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


class RiskAssessmentModel:
    """ML model for risk identification, classification and impact prediction"""
    
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.category_classifier = None
        self.probability_predictor = None
        self.impact_predictor = None
        self.severity_classifier = None
        
        # Preprocessing
        self.scaler = None
        self.text_vectorizer = None
        self.category_encoder = None
        
        self.feature_names = []
        self.risk_categories = ['technical', 'schedule', 'budget', 'resource', 'external', 'quality']
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. ML risk assessment will be disabled.")
    
    def extract_risk_features(self, risks: List[Dict[str, Any]], 
                            project_context: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Extract features from risk data for ML models"""
        
        features_list = []
        
        for risk in risks:
            features = {
                # Text-based features
                'name_length': len(risk.get('name', '')),
                'description_length': len(risk.get('description', '')),
                'has_description': 1 if risk.get('description') else 0,
                
                # Risk content analysis
                'technical_keywords': self._count_technical_keywords(risk),
                'urgency_keywords': self._count_urgency_keywords(risk),
                'cost_keywords': self._count_cost_keywords(risk),
                'time_keywords': self._count_time_keywords(risk),
                'external_keywords': self._count_external_keywords(risk),
                
                # Risk attributes
                'has_mitigation': 1 if risk.get('mitigation_strategy') else 0,
                'has_contingency': 1 if risk.get('contingency_plan') else 0,
                'has_owner': 1 if risk.get('owner') else 0,
                'mitigation_cost': float(risk.get('cost_of_mitigation', 0)),
                
                # Context features from project
                'project_size': self._get_project_size_factor(project_context),
                'project_complexity': self._get_project_complexity(project_context),
                'team_experience': self._get_team_experience_factor(project_context),
                'timeline_pressure': self._get_timeline_pressure(project_context),
                'budget_constraints': self._get_budget_pressure(project_context),
                
                # Historical patterns (simulated)
                'similar_risk_frequency': self._get_similar_risk_frequency(risk),
                'category_risk_factor': self._get_category_risk_factor(risk, project_context)
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def _count_technical_keywords(self, risk: Dict[str, Any]) -> int:
        """Count technical risk indicators in text"""
        keywords = [
            'integration', 'api', 'database', 'performance', 'scalability',
            'security', 'technology', 'system', 'architecture', 'compatibility',
            'legacy', 'migration', 'upgrade', 'infrastructure', 'platform'
        ]
        text = f"{risk.get('name', '')} {risk.get('description', '')}".lower()
        return sum(1 for keyword in keywords if keyword in text)
    
    def _count_urgency_keywords(self, risk: Dict[str, Any]) -> int:
        """Count urgency indicators"""
        keywords = [
            'urgent', 'critical', 'immediate', 'asap', 'deadline',
            'delay', 'behind', 'late', 'overdue', 'rush'
        ]
        text = f"{risk.get('name', '')} {risk.get('description', '')}".lower()
        return sum(1 for keyword in keywords if keyword in text)
    
    def _count_cost_keywords(self, risk: Dict[str, Any]) -> int:
        """Count cost-related risk indicators"""
        keywords = [
            'budget', 'cost', 'expensive', 'funding', 'financial',
            'money', 'price', 'investment', 'expense', 'overrun'
        ]
        text = f"{risk.get('name', '')} {risk.get('description', '')}".lower()
        return sum(1 for keyword in keywords if keyword in text)
    
    def _count_time_keywords(self, risk: Dict[str, Any]) -> int:
        """Count time-related risk indicators"""
        keywords = [
            'schedule', 'timeline', 'duration', 'delay', 'time',
            'milestone', 'deadline', 'sprint', 'phase', 'delivery'
        ]
        text = f"{risk.get('name', '')} {risk.get('description', '')}".lower()
        return sum(1 for keyword in keywords if keyword in text)
    
    def _count_external_keywords(self, risk: Dict[str, Any]) -> int:
        """Count external risk indicators"""
        keywords = [
            'vendor', 'supplier', 'third-party', 'external', 'client',
            'regulatory', 'compliance', 'legal', 'market', 'competitor'
        ]
        text = f"{risk.get('name', '')} {risk.get('description', '')}".lower()
        return sum(1 for keyword in keywords if keyword in text)
    
    def _get_project_size_factor(self, context: Optional[Dict[str, Any]]) -> float:
        """Calculate project size factor"""
        if not context:
            return 2.0
        
        # Use budget as proxy for size
        budget = context.get('total_cost', 0)
        if budget > 1000000:
            return 4.0  # Enterprise
        elif budget > 500000:
            return 3.5  # Large
        elif budget > 100000:
            return 2.5  # Medium
        elif budget > 25000:
            return 2.0  # Small
        else:
            return 1.0  # Micro
    
    def _get_project_complexity(self, context: Optional[Dict[str, Any]]) -> float:
        """Estimate project complexity"""
        if not context:
            return 2.0
        
        complexity_indicators = 0
        description = context.get('description', '').lower()
        
        # Technology complexity
        complex_tech = ['ai', 'machine learning', 'blockchain', 'microservices', 'real-time']
        complexity_indicators += sum(1 for tech in complex_tech if tech in description)
        
        # Integration complexity
        integration_terms = ['integration', 'api', 'third-party', 'legacy']
        complexity_indicators += sum(1 for term in integration_terms if term in description)
        
        # Scale complexity
        if 'enterprise' in description or 'large-scale' in description:
            complexity_indicators += 2
        
        return min(4.0, 1.0 + complexity_indicators * 0.3)
    
    def _get_team_experience_factor(self, context: Optional[Dict[str, Any]]) -> float:
        """Calculate team experience factor"""
        if not context:
            return 2.5
        
        resources = context.get('resources', [])
        if not resources:
            return 2.5
        
        senior_count = 0
        total_count = len(resources)
        
        for resource in resources:
            name = resource.get('name', '').lower()
            if any(level in name for level in ['senior', 'lead', 'principal', 'architect']):
                senior_count += 1
        
        experience_ratio = senior_count / total_count if total_count > 0 else 0.3
        return 1.0 + (experience_ratio * 3.0)  # Scale 1-4
    
    def _get_timeline_pressure(self, context: Optional[Dict[str, Any]]) -> float:
        """Calculate timeline pressure factor"""
        if not context:
            return 2.0
        
        duration = context.get('total_duration', 0)
        complexity = self._get_project_complexity(context)
        
        # Heuristic: if duration seems short for complexity, there's pressure
        expected_duration = complexity * 30  # 30 days per complexity point
        
        if duration > 0 and duration < expected_duration * 0.7:
            return 4.0  # High pressure
        elif duration < expected_duration * 0.9:
            return 3.0  # Medium pressure
        else:
            return 2.0  # Normal
    
    def _get_budget_pressure(self, context: Optional[Dict[str, Any]]) -> float:
        """Calculate budget pressure factor"""
        if not context:
            return 2.0
        
        # Simple heuristic based on cost per day
        total_cost = context.get('total_cost', 0)
        duration = context.get('total_duration', 1)
        
        daily_cost = total_cost / duration if duration > 0 else 0
        
        if daily_cost < 1000:
            return 3.5  # Very tight budget
        elif daily_cost < 3000:
            return 2.5  # Moderate budget
        else:
            return 1.5  # Comfortable budget
    
    def _get_similar_risk_frequency(self, risk: Dict[str, Any]) -> float:
        """Get frequency of similar risks (simulated historical data)"""
        # In real implementation, this would query historical database
        category = risk.get('category', 'technical')
        
        # Simulated frequencies based on category
        frequencies = {
            'technical': 0.7,
            'schedule': 0.8,
            'budget': 0.6,
            'resource': 0.5,
            'external': 0.3,
            'quality': 0.4
        }
        
        return frequencies.get(category, 0.5)
    
    def _get_category_risk_factor(self, risk: Dict[str, Any], context: Optional[Dict[str, Any]]) -> float:
        """Get category-specific risk factor"""
        category = risk.get('category', 'technical')
        
        if not context:
            return 1.0
        
        # Adjust based on project characteristics
        description = context.get('description', '').lower()
        
        if category == 'technical':
            if any(tech in description for tech in ['ai', 'blockchain', 'microservices']):
                return 1.5  # Higher technical risk
            elif 'web application' in description:
                return 0.8  # Lower technical risk
        elif category == 'schedule':
            timeline_pressure = self._get_timeline_pressure(context)
            return timeline_pressure / 2.0  # Scale to 0.5-2.0
        elif category == 'budget':
            budget_pressure = self._get_budget_pressure(context)
            return budget_pressure / 2.0
        
        return 1.0
    
    def generate_training_data(self, num_samples: int = 1500) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data for risk models"""
        
        np.random.seed(42)
        
        risk_data = []
        categories = []
        probabilities = []
        impacts = []
        
        for _ in range(num_samples):
            # Generate risk category
            category = np.random.choice(self.risk_categories, 
                                      p=[0.25, 0.2, 0.15, 0.15, 0.1, 0.15])
            
            # Generate features based on category
            features = self._generate_risk_features_by_category(category)
            
            # Generate probability and impact based on features
            prob = self._calculate_synthetic_probability(features, category)
            impact = self._calculate_synthetic_impact(features, category)
            
            risk_data.append(features)
            categories.append(category)
            probabilities.append(prob)
            impacts.append(impact)
        
        df_features = pd.DataFrame(risk_data)
        return (df_features, 
                np.array(categories), 
                np.array(probabilities), 
                np.array(impacts))
    
    def _generate_risk_features_by_category(self, category: str) -> Dict[str, Any]:
        """Generate realistic features for a given risk category"""
        
        base_features = {
            'name_length': np.random.randint(20, 100),
            'description_length': np.random.randint(50, 300),
            'has_description': 1,
            'has_mitigation': np.random.choice([0, 1], p=[0.3, 0.7]),
            'has_contingency': np.random.choice([0, 1], p=[0.5, 0.5]),
            'has_owner': np.random.choice([0, 1], p=[0.2, 0.8]),
            'mitigation_cost': np.random.exponential(5000),
            'project_size': np.random.uniform(1.0, 4.0),
            'project_complexity': np.random.uniform(1.0, 4.0),
            'team_experience': np.random.uniform(1.0, 4.0),
            'timeline_pressure': np.random.uniform(1.0, 4.0),
            'budget_constraints': np.random.uniform(1.0, 4.0),
            'similar_risk_frequency': np.random.uniform(0.1, 0.9)
        }
        
        # Category-specific keyword counts
        if category == 'technical':
            base_features.update({
                'technical_keywords': np.random.randint(2, 8),
                'urgency_keywords': np.random.randint(0, 3),
                'cost_keywords': np.random.randint(0, 2),
                'time_keywords': np.random.randint(0, 3),
                'external_keywords': np.random.randint(0, 1),
                'category_risk_factor': np.random.uniform(0.8, 1.5)
            })
        elif category == 'schedule':
            base_features.update({
                'technical_keywords': np.random.randint(0, 2),
                'urgency_keywords': np.random.randint(3, 8),
                'cost_keywords': np.random.randint(0, 2),
                'time_keywords': np.random.randint(4, 10),
                'external_keywords': np.random.randint(0, 2),
                'category_risk_factor': np.random.uniform(1.0, 2.0)
            })
        elif category == 'budget':
            base_features.update({
                'technical_keywords': np.random.randint(0, 1),
                'urgency_keywords': np.random.randint(1, 4),
                'cost_keywords': np.random.randint(4, 10),
                'time_keywords': np.random.randint(0, 3),
                'external_keywords': np.random.randint(0, 3),
                'category_risk_factor': np.random.uniform(0.8, 1.8)
            })
        elif category == 'external':
            base_features.update({
                'technical_keywords': np.random.randint(0, 2),
                'urgency_keywords': np.random.randint(0, 3),
                'cost_keywords': np.random.randint(1, 4),
                'time_keywords': np.random.randint(1, 4),
                'external_keywords': np.random.randint(3, 8),
                'category_risk_factor': np.random.uniform(0.5, 2.0)
            })
        else:  # resource, quality
            base_features.update({
                'technical_keywords': np.random.randint(0, 3),
                'urgency_keywords': np.random.randint(1, 5),
                'cost_keywords': np.random.randint(1, 4),
                'time_keywords': np.random.randint(1, 5),
                'external_keywords': np.random.randint(0, 2),
                'category_risk_factor': np.random.uniform(0.7, 1.3)
            })
        
        return base_features
    
    def _calculate_synthetic_probability(self, features: Dict[str, Any], category: str) -> int:
        """Calculate synthetic probability based on features"""
        
        base_prob = 3.0  # Medium probability
        
        # Adjust based on features
        if features['timeline_pressure'] > 3.0:
            base_prob += 0.5
        if features['project_complexity'] > 3.0:
            base_prob += 0.3
        if features['team_experience'] < 2.0:
            base_prob += 0.4
        if features['similar_risk_frequency'] > 0.6:
            base_prob += 0.3
        
        # Category-specific adjustments
        category_adjustments = {
            'schedule': 0.2,
            'technical': 0.1,
            'resource': -0.1,
            'external': -0.2,
            'budget': 0.0,
            'quality': 0.0
        }
        
        base_prob += category_adjustments.get(category, 0)
        
        # Add noise and clamp to 1-5
        prob = base_prob + np.random.normal(0, 0.3)
        return int(np.clip(prob, 1, 5))
    
    def _calculate_synthetic_impact(self, features: Dict[str, Any], category: str) -> int:
        """Calculate synthetic impact based on features"""
        
        base_impact = 3.0  # Medium impact
        
        # Adjust based on features
        if features['project_size'] > 3.0:
            base_impact += 0.4
        if features['budget_constraints'] > 3.0:
            base_impact += 0.3
        if features['mitigation_cost'] > 10000:
            base_impact += 0.2
        
        # Category-specific adjustments
        category_adjustments = {
            'budget': 0.3,
            'schedule': 0.2,
            'quality': 0.1,
            'technical': 0.0,
            'resource': -0.1,
            'external': 0.1
        }
        
        base_impact += category_adjustments.get(category, 0)
        
        # Add noise and clamp to 1-5
        impact = base_impact + np.random.normal(0, 0.3)
        return int(np.clip(impact, 1, 5))
    
    def train_models(self, features_df: Optional[pd.DataFrame] = None,
                    categories: Optional[np.ndarray] = None,
                    probabilities: Optional[np.ndarray] = None,
                    impacts: Optional[np.ndarray] = None):
        """Train risk assessment models"""
        
        if not SKLEARN_AVAILABLE:
            logger.error("Cannot train models: scikit-learn not available")
            return False
        
        # Use provided data or generate synthetic data
        if any(x is None for x in [features_df, categories, probabilities, impacts]):
            logger.info("Generating synthetic training data for risk models")
            features_df, categories, probabilities, impacts = self.generate_training_data()
        
        # Store feature names
        self.feature_names = features_df.columns.tolist()
        X = features_df.values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode categories
        self.category_encoder = LabelEncoder()
        y_categories = self.category_encoder.fit_transform(categories)
        
        # Train category classifier
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X_scaled, y_categories, test_size=0.2, random_state=42
        )
        
        self.category_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.category_classifier.fit(X_train, y_cat_train)
        
        # Evaluate category classifier
        cat_pred = self.category_classifier.predict(X_test)
        cat_accuracy = accuracy_score(y_cat_test, cat_pred)
        logger.info(f"Category classifier accuracy: {cat_accuracy:.3f}")
        
        # Train probability predictor
        self.probability_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.probability_predictor.fit(X_train, probabilities[:-len(X_test)])
        prob_pred = self.probability_predictor.predict(X_test)
        prob_mse = mean_squared_error(probabilities[-len(X_test):], prob_pred)
        logger.info(f"Probability predictor MSE: {prob_mse:.3f}")
        
        # Train impact predictor
        self.impact_predictor = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.impact_predictor.fit(X_train, impacts[:-len(X_test)])
        impact_pred = self.impact_predictor.predict(X_test)
        impact_mse = mean_squared_error(impacts[-len(X_test):], impact_pred)
        logger.info(f"Impact predictor MSE: {impact_mse:.3f}")
        
        # Save models
        self.save_models()
        return True
    
    def predict_risk_assessment(self, risks: List[Dict[str, Any]], 
                               project_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Predict risk categories, probabilities, and impacts"""
        
        if not SKLEARN_AVAILABLE or not self._models_ready():
            logger.warning("ML models not available, using fallback assessment")
            return self._fallback_risk_assessment(risks)
        
        try:
            # Extract features
            features_df = self.extract_risk_features(risks, project_context)
            
            # Ensure feature order matches training
            if self.feature_names:
                features_df = features_df.reindex(columns=self.feature_names, fill_value=0)
            
            # Scale features
            X_scaled = self.scaler.transform(features_df.values)
            
            # Make predictions
            category_pred = self.category_classifier.predict(X_scaled)
            category_proba = self.category_classifier.predict_proba(X_scaled)
            prob_pred = self.probability_predictor.predict(X_scaled)
            impact_pred = self.impact_predictor.predict(X_scaled)
            
            # Process results
            results = []
            for i, risk in enumerate(risks):
                predicted_category = self.category_encoder.inverse_transform([category_pred[i]])[0]
                category_confidence = np.max(category_proba[i])
                
                results.append({
                    'risk_id': risk.get('id', f'risk_{i}'),
                    'predicted_category': predicted_category,
                    'category_confidence': float(category_confidence),
                    'predicted_probability': max(1, min(5, int(round(prob_pred[i])))),
                    'predicted_impact': max(1, min(5, int(round(impact_pred[i])))),
                    'risk_score': max(1, min(5, int(round(prob_pred[i])))) * max(1, min(5, int(round(impact_pred[i])))),
                    'assessment_method': 'ml_model'
                })
            
            logger.info(f"ML risk assessment completed for {len(risks)} risks")
            return results
            
        except Exception as e:
            logger.error(f"ML risk prediction failed: {e}")
            return self._fallback_risk_assessment(risks)
    
    def _models_ready(self) -> bool:
        """Check if all models are trained and ready"""
        return all([
            self.category_classifier,
            self.probability_predictor, 
            self.impact_predictor,
            self.scaler
        ])
    
    def _fallback_risk_assessment(self, risks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback risk assessment when ML is not available"""
        
        results = []
        for i, risk in enumerate(risks):
            # Use existing values or apply simple heuristics
            category = risk.get('category', 'technical')
            probability = risk.get('probability', 3)
            impact = risk.get('impact', 3)
            
            results.append({
                'risk_id': risk.get('id', f'risk_{i}'),
                'predicted_category': category,
                'category_confidence': 0.7,
                'predicted_probability': probability,
                'predicted_impact': impact,
                'risk_score': probability * impact,
                'assessment_method': 'heuristic'
            })
        
        return results
    
    def save_models(self):
        """Save trained models to disk"""
        
        if not self._models_ready():
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_data = {
            'category_classifier': self.category_classifier,
            'probability_predictor': self.probability_predictor,
            'impact_predictor': self.impact_predictor,
            'scaler': self.scaler,
            'category_encoder': self.category_encoder,
            'feature_names': self.feature_names,
            'timestamp': timestamp
        }
        
        model_path = self.models_dir / f"risk_model_{timestamp}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save latest model
        latest_path = self.models_dir / "risk_model_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Risk models saved to {model_path}")
    
    def load_models(self, model_path: Optional[str] = None):
        """Load pre-trained models from disk"""
        
        if model_path is None:
            model_path = self.models_dir / "risk_model_latest.pkl"
        else:
            model_path = Path(model_path)
        
        if not model_path.exists():
            logger.warning(f"Risk model file not found: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.category_classifier = model_data['category_classifier']
            self.probability_predictor = model_data['probability_predictor']
            self.impact_predictor = model_data['impact_predictor']
            self.scaler = model_data['scaler']
            self.category_encoder = model_data['category_encoder']
            self.feature_names = model_data.get('feature_names', [])
            
            logger.info(f"Risk models loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load risk models: {e}")
            return False