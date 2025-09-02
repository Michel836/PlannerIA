"""
Test Suite for Plan Validator
Tests for Pydantic models, JSON schema validation, and plan correction functionality
"""

import pytest
import json
import tempfile
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.project_planner.agents.validator import (
    PlanValidator,
    ProjectOverview,
    Task,
    Phase,
    WBS,
    Dependency,
    Risk,
    Resource,
    Milestone,
    RAGCitation,
    Documentation,
    Metadata,
    ProjectPlan
)


class TestPydanticModels:
    """Test Pydantic model validation"""
    
    def test_project_overview_valid(self):
        """Test valid project overview creation"""
        overview = ProjectOverview(
            title="Test Project",
            description="A test project for validation",
            objectives=["Objective 1", "Objective 2"],
            success_criteria=["Criteria 1"],
            total_duration=30.0,
            total_cost=50000.0,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31)
        )
        
        assert overview.title == "Test Project"
        assert overview.total_duration == 30.0
        assert overview.total_cost == 50000.0
        assert len(overview.objectives) == 2
    
    def test_project_overview_title_too_long(self):
        """Test project overview with title too long"""
        with pytest.raises(ValueError):
            ProjectOverview(
                title="x" * 201,  # Exceeds max length of 200
                description="Valid description"
            )
    
    def test_project_overview_negative_cost(self):
        """Test project overview with negative cost"""
        with pytest.raises(ValueError):
            ProjectOverview(
                title="Test Project",
                description="Valid description",
                total_cost=-1000.0  # Negative cost should be invalid
            )
    
    def test_task_valid(self):
        """Test valid task creation"""
        task = Task(
            id="task_001",
            name="Development Task",
            description="Develop feature X",
            duration=5.0,
            duration_unit="days",
            effort=40.0,
            cost=4000.0,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 6),
            status="not_started",
            priority="high",
            assigned_resources=["dev1", "dev2"],
            deliverables=["Feature X", "Documentation"]
        )
        
        assert task.id == "task_001"
        assert task.duration == 5.0
        assert task.status == "not_started"
        assert task.priority == "high"
        assert len(task.assigned_resources) == 2
    
    def test_task_invalid_status(self):
        """Test task with invalid status"""
        with pytest.raises(ValueError):
            Task(
                id="task_001",
                name="Test Task",
                duration=1.0,
                status="invalid_status"  # Should match regex pattern
            )
    
    def test_task_negative_duration(self):
        """Test task with negative duration"""
        with pytest.raises(ValueError):
            Task(
                id="task_001",
                name="Test Task",
                duration=-1.0  # Negative duration should be invalid
            )
    
    def test_risk_valid(self):
        """Test valid risk creation"""
        risk = Risk(
            id="risk_001",
            name="Technical Risk",
            description="Integration complexity",
            category="technical",
            probability=3,
            impact=4,
            mitigation_strategy="Use proven technologies",
            contingency_plan="Fallback to simpler solution",
            owner="Tech Lead",
            status="identified"
        )
        
        assert risk.id == "risk_001"
        assert risk.probability == 3
        assert risk.impact == 4
        assert risk.risk_score == 12  # Should auto-calculate
        assert risk.category == "technical"
    
    def test_risk_probability_out_of_range(self):
        """Test risk with probability out of valid range"""
        with pytest.raises(ValueError):
            Risk(
                id="risk_001",
                name="Test Risk",
                probability=6,  # Should be 1-5
                impact=3
            )
    
    def test_risk_auto_calculate_score(self):
        """Test risk score auto-calculation"""
        risk = Risk(
            id="risk_001",
            name="Test Risk",
            probability=4,
            impact=3
        )
        
        assert risk.risk_score == 12  # 4 * 3
    
    def test_dependency_valid(self):
        """Test valid dependency creation"""
        dependency = Dependency(
            predecessor="task_001",
            successor="task_002",
            type="finish_to_start",
            lag=2.0
        )
        
        assert dependency.predecessor == "task_001"
        assert dependency.successor == "task_002"
        assert dependency.type == "finish_to_start"
        assert dependency.lag == 2.0
    
    def test_dependency_invalid_type(self):
        """Test dependency with invalid type"""
        with pytest.raises(ValueError):
            Dependency(
                predecessor="task_001",
                successor="task_002",
                type="invalid_type"  # Should match regex pattern
            )


class TestPlanValidator:
    """Test PlanValidator class functionality"""
    
    @pytest.fixture
    def validator(self):
        """Create a validator instance for testing"""
        # Create temporary schema file
        schema_content = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["run_id", "timestamp", "project_overview"],
            "properties": {
                "run_id": {"type": "string"},
                "timestamp": {"type": "string"},
                "project_overview": {
                    "type": "object",
                    "required": ["title", "description"],
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_content, f)
            schema_path = f.name
        
        return PlanValidator(schema_path)
    
    @pytest.fixture
    def valid_plan_data(self):
        """Create valid plan data for testing"""
        return {
            "run_id": "test-run-001",
            "timestamp": datetime.now().isoformat(),
            "brief": "Test project brief",
            "project_overview": {
                "title": "Test Project",
                "description": "A comprehensive test project"
            },
            "wbs": {
                "phases": [
                    {
                        "id": "phase_1",
                        "name": "Development Phase",
                        "tasks": [
                            {
                                "id": "task_1",
                                "name": "Development Task",
                                "duration": 5.0
                            }
                        ]
                    }
                ]
            },
            "tasks": [],
            "dependencies": [],
            "risks": [],
            "resources": [],
            "milestones": [],
            "critical_path": [],
            "rag_citations": [],
            "documentation": {},
            "metadata": {
                "model_used": "gpt-5",
                "validation_status": "pending"
            }
        }
    
    @pytest.fixture
    def invalid_plan_data(self):
        """Create invalid plan data for testing corrections"""
        return {
            "run_id": "test-run-002",
            "timestamp": "invalid-timestamp",
            "project_overview": {
                "title": "x" * 300,  # Too long
                "description": "",  # Empty
                "total_duration": -10,  # Negative
                "total_cost": "not_a_number"  # Wrong type
            },
            "tasks": [
                {
                    "id": "task_1",
                    "name": "Test Task",
                    "duration": "five",  # Wrong type
                    "priority": "super_high",  # Invalid value
                    "status": "maybe_started"  # Invalid value
                }
            ],
            "risks": [
                {
                    "id": "risk_1",
                    "name": "Test Risk",
                    "probability": 10,  # Out of range
                    "impact": 0  # Out of range
                }
            ]
        }
    
    def test_validate_valid_plan(self, validator, valid_plan_data):
        """Test validation of a valid plan"""
        result = validator.validate_plan(valid_plan_data)
        
        assert result is not None
        assert result['run_id'] == 'test-run-001'
        assert result['metadata']['validation_status'] == 'valid'
        assert 'project_overview' in result
        assert result['project_overview']['title'] == 'Test Project'
    
    def test_validate_plan_with_corrections(self, validator, invalid_plan_data):
        """Test validation with automatic corrections"""
        result = validator.validate_plan(invalid_plan_data)
        
        assert result is not None
        assert result['metadata']['validation_status'] in ['corrected', 'failed']
        
        # Check that required fields are present
        assert 'run_id' in result
        assert 'timestamp' in result
        assert 'project_overview' in result
    
    def test_validate_plan_missing_required_fields(self, validator):
        """Test validation with missing required fields"""
        incomplete_data = {
            "run_id": "test-run-003"
            # Missing timestamp and project_overview
        }
        
        result = validator.validate_plan(incomplete_data)
        
        assert result is not None
        assert 'timestamp' in result  # Should be auto-added
        assert 'project_overview' in result  # Should be auto-added
        assert result['metadata']['validation_status'] in ['corrected', 'failed']
    
    def test_validate_empty_plan(self, validator):
        """Test validation of completely empty plan"""
        empty_data = {}
        
        result = validator.validate_plan(empty_data)
        
        assert result is not None
        assert 'run_id' in result
        assert 'timestamp' in result
        assert 'project_overview' in result
        assert result['metadata']['validation_status'] == 'failed'
    
    def test_validate_tasks(self, validator):
        """Test individual task validation"""
        valid_tasks = [
            {
                "id": "task_1",
                "name": "Valid Task",
                "duration": 3.0,
                "priority": "medium",
                "status": "not_started"
            },
            {
                "id": "task_2",
                "name": "Another Task",
                "duration": 2.5,
                "priority": "high"
            }
        ]
        
        result = validator.validate_tasks(valid_tasks)
        
        assert len(result) == 2
        assert result[0]['id'] == 'task_1'
        assert result[0]['duration'] == 3.0
        assert result[1]['id'] == 'task_2'
    
    def test_validate_tasks_with_missing_fields(self, validator):
        """Test task validation with missing required fields"""
        incomplete_tasks = [
            {
                "name": "Task without ID",
                # Missing required 'id' field
            },
            {
                "id": "task_2"
                # Missing required 'name' and 'duration' fields
            }
        ]
        
        result = validator.validate_tasks(incomplete_tasks)
        
        assert len(result) == 2
        # Check that missing fields were auto-generated
        assert 'id' in result[0]
        assert 'name' in result[0]
        assert 'duration' in result[0]
        assert result[0]['duration'] >= 1.0  # Should have default duration
        
        assert result[1]['id'] == 'task_2'
        assert 'name' in result[1]
        assert 'duration' in result[1]
    
    def test_validate_risks(self, validator):
        """Test individual risk validation"""
        valid_risks = [
            {
                "id": "risk_1",
                "name": "Technical Risk",
                "probability": 3,
                "impact": 4,
                "category": "technical"
            },
            {
                "id": "risk_2",
                "name": "Schedule Risk",
                "probability": 2,
                "impact": 3,
                "category": "schedule"
            }
        ]
        
        result = validator.validate_risks(valid_risks)
        
        assert len(result) == 2
        assert result[0]['risk_score'] == 12  # 3 * 4
        assert result[1]['risk_score'] == 6   # 2 * 3
        assert result[0]['category'] == 'technical'
    
    def test_validate_risks_with_invalid_values(self, validator):
        """Test risk validation with out-of-range values"""
        invalid_risks = [
            {
                "id": "risk_1",
                "name": "Invalid Risk",
                "probability": 10,  # Out of range (1-5)
                "impact": 0,       # Out of range (1-5)
                "category": "technical"
            }
        ]
        
        result = validator.validate_risks(invalid_risks)
        
        assert len(result) == 1
        # Values should be clamped to valid range
        assert 1 <= result[0]['probability'] <= 5
        assert 1 <= result[0]['impact'] <= 5
        assert result[0]['risk_score'] == result[0]['probability'] * result[0]['impact']
    
    def test_create_minimal_valid_plan(self, validator):
        """Test creation of minimal valid plan from invalid data"""
        invalid_data = {
            "some_random_field": "random_value",
            "another_field": 12345
        }
        
        result = validator._create_minimal_valid_plan(invalid_data)
        
        # Check that all required fields are present
        assert 'run_id' in result
        assert 'timestamp' in result
        assert 'project_overview' in result
        assert 'wbs' in result
        assert result['metadata']['validation_status'] == 'failed'
        
        # Check basic structure
        assert 'title' in result['project_overview']
        assert 'description' in result['project_overview']
        assert 'phases' in result['wbs']


class TestValidationScenarios:
    """Test various validation scenarios and edge cases"""
    
    def test_plan_with_circular_dependencies(self):
        """Test plan with circular task dependencies"""
        validator = PlanValidator()
        
        plan_data = {
            "run_id": "circular-test",
            "timestamp": datetime.now().isoformat(),
            "project_overview": {
                "title": "Circular Dependency Test",
                "description": "Testing circular dependencies"
            },
            "wbs": {"phases": []},
            "dependencies": [
                {
                    "predecessor": "task_a",
                    "successor": "task_b",
                    "type": "finish_to_start"
                },
                {
                    "predecessor": "task_b", 
                    "successor": "task_a",  # Creates circular dependency
                    "type": "finish_to_start"
                }
            ]
        }
        
        result = validator.validate_plan(plan_data)
        
        assert result is not None
        # Should still validate but may have warnings in metadata
        assert 'dependencies' in result
    
    def test_plan_with_unicode_content(self):
        """Test plan validation with Unicode content"""
        validator = PlanValidator()
        
        plan_data = {
            "run_id": "unicode-test",
            "timestamp": datetime.now().isoformat(),
            "project_overview": {
                "title": "ÐŸÑ€Ð¾ÐµÐºÑ‚ Ñ Unicode ÑÐ¸Ð¼Ð²Ð¾Ð»Ð°Ð¼Ð¸ ðŸš€",
                "description": "Testing Unicode: franÃ§ais, espaÃ±ol, ä¸­æ–‡, Ø§Ù„Ø¹Ø±Ø¨ÙŠØ©"
            },
            "wbs": {"phases": []},
            "tasks": [
                {
                    "id": "task_unicode",
                    "name": "TÃ¢che avec accents Ã© Ã  Ã§",
                    "description": "Description avec Ã©mojis ðŸ˜€ âœ… ðŸ”§",
                    "duration": 2.0
                }
            ]
        }
        
        result = validator.validate_plan(plan_data)
        
        assert result is not None
        assert "Unicode" in result['project_overview']['title']
        assert "Ã©mojis" in result['tasks'][0]['description']
    
    def test_plan_with_extreme_values(self):
        """Test plan validation with extreme numerical values"""
        validator = PlanValidator()
        
        plan_data = {
            "run_id": "extreme-values-test",
            "timestamp": datetime.now().isoformat(),
            "project_overview": {
                "title": "Extreme Values Test",
                "description": "Testing extreme numerical values",
                "total_duration": 999999.99,
                "total_cost": 1000000000.00
            },
            "wbs": {"phases": []},
            "tasks": [
                {
                    "id": "extreme_task",
                    "name": "Extreme Task",
                    "duration": 0.001,  # Very small duration
                    "cost": 999999999.99  # Very large cost
                }
            ]
        }
        
        result = validator.validate_plan(plan_data)
        
        assert result is not None
        assert result['project_overview']['total_duration'] == 999999.99
        assert result['tasks'][0]['duration'] == 0.001
    
    def test_plan_with_nested_wbs_structure(self):
        """Test validation of complex nested WBS structure"""
        validator = PlanValidator()
        
        plan_data = {
            "run_id": "nested-wbs-test",
            "timestamp": datetime.now().isoformat(),
            "project_overview": {
                "title": "Nested WBS Test",
                "description": "Testing complex WBS structure"
            },
            "wbs": {
                "phases": [
                    {
                        "id": "phase_1",
                        "name": "Phase 1",
                        "description": "First phase",
                        "tasks": [
                            {
                                "id": "task_1_1",
                                "name": "Task 1.1",
                                "duration": 5.0,
                                "deliverables": ["Deliverable A", "Deliverable B"]
                            },
                            {
                                "id": "task_1_2", 
                                "name": "Task 1.2",
                                "duration": 3.0,
                                "assigned_resources": ["resource_1", "resource_2"]
                            }
                        ]
                    },
                    {
                        "id": "phase_2",
                        "name": "Phase 2",
                        "tasks": [
                            {
                                "id": "task_2_1",
                                "name": "Task 2.1",
                                "duration": 7.0,
                                "priority": "critical"
                            }
                        ]
                    }
                ]
            }
        }
        
        result = validator.validate_plan(plan_data)
        
        assert result is not None
        assert len(result['wbs']['phases']) == 2
        assert len(result['wbs']['phases'][0]['tasks']) == 2
        assert len(result['wbs']['phases'][1]['tasks']) == 1
        assert result['wbs']['phases'][0]['tasks'][0]['id'] == 'task_1_1'


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_validator_with_missing_schema_file(self):
        """Test validator creation with missing schema file"""
        validator = PlanValidator("nonexistent_schema.json")
        
        # Should still work but without schema validation
        assert validator.schema == {}
    
    def test_validate_plan_with_none_input(self):
        """Test validation with None input"""
        validator = PlanValidator()
        
        with pytest.raises(TypeError):
            validator.validate_plan(None)
    
    def test_validate_plan_with_invalid_json_structure(self):
        """Test validation with invalid JSON structure"""
        validator = PlanValidator()
        
        # Test with non-dict input
        result = validator.validate_plan("not a dictionary")
        
        # Should handle gracefully and create minimal plan
        assert result is not None
        assert result['metadata']['validation_status'] == 'failed'
    
    def test_validate_tasks_with_none_input(self):
        """Test task validation with None input"""
        validator = PlanValidator()
        
        result = validator.validate_tasks(None)
        
        assert result == []
    
    def test_validate_risks_with_empty_list(self):
        """Test risk validation with empty list"""
        validator = PlanValidator()
        
        result = validator.validate_risks([])
        
        assert result == []


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
