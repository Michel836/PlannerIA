"""
Project Optimizer - Optimizes project plans using NetworkX and various algorithms
"""

import logging
import math
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, timedelta

# Optional NetworkX import with fallback
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False

# Configure logging
logger = logging.getLogger(__name__)


class ProjectOptimizer:
    """Optimizes project plans using graph algorithms and heuristics"""
    
    def __init__(self):
        self.graph = None
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available. Critical path analysis will be disabled.")
    
    def optimize_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize a project plan with multiple techniques
        
        Args:
            plan: Validated project plan
            
        Returns:
            Optimized project plan
        """
        if not isinstance(plan, dict):
            logger.error("Invalid plan data provided to optimizer")
            return plan
        
        optimized_plan = plan.copy()
        
        try:
            # Extract tasks and dependencies
            tasks = self._extract_tasks(plan)
            dependencies = plan.get('dependencies', [])
            
            if not tasks:
                logger.info("No tasks found in plan, skipping optimization")
                return optimized_plan
            
            # Apply optimizations based on available libraries
            if HAS_NETWORKX and tasks:
                # Build project graph
                self.graph = self._build_project_graph(tasks, dependencies)
                
                # Apply graph-based optimizations
                optimized_plan = self._optimize_critical_path(optimized_plan)
            else:
                logger.info("NetworkX not available, using basic optimization")
                optimized_plan = self._basic_optimization(optimized_plan)
            
            # Apply general optimizations (don't require NetworkX)
            optimized_plan = self._optimize_resource_allocation(optimized_plan)
            optimized_plan = self._calculate_project_metrics(optimized_plan)
            
            # Update metadata
            metadata = optimized_plan.get('metadata', {})
            metadata.update({
                'optimized': True,
                'optimization_date': datetime.now().isoformat(),
                'optimizer_version': '1.0',
                'networkx_available': HAS_NETWORKX
            })
            optimized_plan['metadata'] = metadata
            
            logger.info("Plan optimization completed successfully")
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Return original plan if optimization fails
            metadata = optimized_plan.get('metadata', {})
            metadata['optimization_error'] = str(e)
            optimized_plan['metadata'] = metadata
        
        return optimized_plan
    
    def _extract_tasks(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all tasks from the plan structure"""
        tasks = []
        
        # Direct tasks list
        if plan.get('tasks') and isinstance(plan['tasks'], list):
            tasks.extend([t for t in plan['tasks'] if isinstance(t, dict)])
        
        # Tasks from WBS phases
        wbs = plan.get('wbs', {})
        if isinstance(wbs, dict) and wbs.get('phases'):
            phases = wbs['phases']
            if isinstance(phases, list):
                for phase in phases:
                    if isinstance(phase, dict) and phase.get('tasks'):
                        phase_tasks = phase['tasks']
                        if isinstance(phase_tasks, list):
                            tasks.extend([t for t in phase_tasks if isinstance(t, dict)])
        
        # Ensure tasks have required fields
        validated_tasks = []
        for i, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            
            # Ensure task has ID
            if not task.get('id'):
                task['id'] = f"task_{i+1}"
            
            # Ensure task has duration
            if 'duration' not in task:
                task['duration'] = 1.0
            else:
                try:
                    task['duration'] = float(task['duration'])
                except (ValueError, TypeError):
                    task['duration'] = 1.0
            
            validated_tasks.append(task)
        
        return validated_tasks
    
    def _build_project_graph(self, tasks: List[Dict[str, Any]], dependencies: List[Dict[str, Any]]) -> Optional[nx.DiGraph]:
        """Build a directed graph representation of the project"""
        if not HAS_NETWORKX:
            return None
        
        try:
            graph = nx.DiGraph()
            
            # Add task nodes
            for task in tasks:
                task_id = task.get('id', '')
                if task_id:
                    graph.add_node(task_id, **task)
            
            # Add dependency edges
            if isinstance(dependencies, list):
                for dep in dependencies:
                    if not isinstance(dep, dict):
                        continue
                    
                    predecessor = dep.get('predecessor', '')
                    successor = dep.get('successor', '')
                    dep_type = dep.get('type', 'finish_to_start')
                    lag = dep.get('lag', 0)
                    
                    # Validate lag
                    try:
                        lag = float(lag)
                    except (ValueError, TypeError):
                        lag = 0
                    
                    if (predecessor and successor and 
                        predecessor in graph and successor in graph):
                        graph.add_edge(predecessor, successor, 
                                     type=dep_type, lag=lag)
            
            return graph
        
        except Exception as e:
            logger.error(f"Failed to build project graph: {e}")
            return None
    
    def _basic_optimization(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Basic optimization without NetworkX"""
        try:
            tasks = self._extract_tasks(plan)
            
            # Simple critical path approximation
            # Sort by priority and dependencies (basic heuristic)
            high_priority_tasks = [t for t in tasks if t.get('priority') in ['high', 'critical']]
            long_duration_tasks = [t for t in tasks if t.get('duration', 0) > 5]
            
            # Combine into approximate critical path
            critical_candidates = set()
            critical_candidates.update(t['id'] for t in high_priority_tasks if t.get('id'))
            critical_candidates.update(t['id'] for t in long_duration_tasks if t.get('id'))
            
            plan['critical_path'] = list(critical_candidates)[:10]  # Limit size
            
            # Mark critical tasks
            self._mark_critical_tasks(plan, plan['critical_path'])
            
            return plan
        
        except Exception as e:
            logger.error(f"Basic optimization failed: {e}")
            return plan
    
    def _optimize_critical_path(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Find and optimize the critical path"""
        if not self.graph or not HAS_NETWORKX:
            return self._basic_optimization(plan)
        
        try:
            # Calculate critical path using longest path algorithm
            critical_path = self._find_critical_path()
            
            if critical_path:
                plan['critical_path'] = critical_path
                
                # Calculate critical path duration
                total_duration = self._calculate_critical_path_duration(critical_path)
                
                # Update project overview
                overview = plan.get('project_overview', {})
                overview.update({
                    'critical_path_duration': total_duration,
                    'critical_path_tasks': len(critical_path)
                })
                plan['project_overview'] = overview
                
                # Mark critical tasks
                self._mark_critical_tasks(plan, critical_path)
                
                logger.info(f"Critical path found with {len(critical_path)} tasks and {total_duration} duration")
        
        except Exception as e:
            logger.warning(f"Critical path calculation failed: {e}")
            # Fall back to basic optimization
            plan = self._basic_optimization(plan)
        
        return plan
    
    def _find_critical_path(self) -> List[str]:
        """Find the critical path using topological sort and longest path"""
        if not self.graph or not HAS_NETWORKX:
            return []
        
        try:
            # Check if graph is acyclic
            if not nx.is_directed_acyclic_graph(self.graph):
                logger.warning("Project contains cycles, attempting to break them")
                self.graph = self._break_cycles(self.graph)
            
            if not self.graph.nodes():
                return []
            
            # Calculate earliest start times using dynamic programming
            earliest_start = {}
            earliest_finish = {}
            
            # Process nodes in topological order
            try:
                topo_order = list(nx.topological_sort(self.graph))
            except nx.NetworkXError:
                logger.error("Cannot determine topological order")
                return []
            
            for node in topo_order:
                node_data = self.graph.nodes[node]
                duration = float(node_data.get('duration', 1.0))
                
                # Calculate earliest start time
                pred_finish_times = []
                for pred in self.graph.predecessors(node):
                    if pred in earliest_finish:
                        edge_data = self.graph.edges[pred, node]
                        lag = float(edge_data.get('lag', 0))
                        pred_finish_times.append(earliest_finish[pred] + lag)
                
                earliest_start[node] = max(pred_finish_times) if pred_finish_times else 0
                earliest_finish[node] = earliest_start[node] + duration
            
            # Find the longest path (critical path)
            if not earliest_finish:
                return []
            
            # Start from the node with latest finish time
            end_nodes = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]
            if not end_nodes:
                end_node = max(earliest_finish, key=earliest_finish.get)
            else:
                end_node = max(end_nodes, key=lambda n: earliest_finish.get(n, 0))
            
            # Backtrack to find the critical path
            critical_path = []
            current = end_node
            visited = set()
            
            while current and current not in visited:
                visited.add(current)
                critical_path.insert(0, current)
                
                # Find predecessor on critical path
                best_pred = None
                target_start = earliest_start.get(current, 0)
                
                for pred in self.graph.predecessors(current):
                    if pred in earliest_finish:
                        edge_data = self.graph.edges[pred, current]
                        lag = float(edge_data.get('lag', 0))
                        
                        if abs(earliest_finish[pred] + lag - target_start) < 0.01:
                            best_pred = pred
                            break
                
                current = best_pred
            
            return critical_path
        
        except Exception as e:
            logger.error(f"Critical path calculation error: {e}")
            return []
    
    def _break_cycles(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Break cycles in the project graph"""
        if not HAS_NETWORKX:
            return graph
        
        try:
            # Create a copy to avoid modifying original
            graph_copy = graph.copy()
            
            # Find and break cycles
            cycles = list(nx.simple_cycles(graph_copy))
            edges_removed = 0
            
            for cycle in cycles:
                if len(cycle) > 1:
                    # Remove the edge with minimum weight (shortest duration)
                    min_weight = float('inf')
                    edge_to_remove = None
                    
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        
                        if graph_copy.has_edge(u, v):
                            weight = graph_copy.nodes[u].get('duration', 1.0)
                            if weight < min_weight:
                                min_weight = weight
                                edge_to_remove = (u, v)
                    
                    if edge_to_remove and graph_copy.has_edge(*edge_to_remove):
                        graph_copy.remove_edge(*edge_to_remove)
                        edges_removed += 1
                        logger.info(f"Removed cyclic dependency: {edge_to_remove[0]} -> {edge_to_remove[1]}")
            
            if edges_removed > 0:
                logger.info(f"Removed {edges_removed} cyclic dependencies")
            
            return graph_copy
        
        except Exception as e:
            logger.error(f"Cycle breaking failed: {e}")
            return graph
    
    def _calculate_critical_path_duration(self, critical_path: List[str]) -> float:
        """Calculate total duration of the critical path"""
        if not critical_path or not self.graph:
            return 0.0
        
        total_duration = 0.0
        
        for node in critical_path:
            if node in self.graph:
                duration = self.graph.nodes[node].get('duration', 1.0)
                try:
                    total_duration += float(duration)
                except (ValueError, TypeError):
                    total_duration += 1.0
        
        return round(total_duration, 2)
    
    def _mark_critical_tasks(self, plan: Dict[str, Any], critical_path: List[str]):
        """Mark tasks that are on the critical path"""
        if not critical_path:
            return
        
        critical_set = set(critical_path)
        
        # Mark tasks in direct task list
        tasks = plan.get('tasks', [])
        if isinstance(tasks, list):
            for task in tasks:
                if isinstance(task, dict):
                    task_id = task.get('id', '')
                    task['is_critical'] = task_id in critical_set
        
        # Mark tasks in WBS phases
        wbs = plan.get('wbs', {})
        if isinstance(wbs, dict) and wbs.get('phases'):
            phases = wbs['phases']
            if isinstance(phases, list):
                for phase in phases:
                    if isinstance(phase, dict) and phase.get('tasks'):
                        phase_tasks = phase['tasks']
                        if isinstance(phase_tasks, list):
                            for task in phase_tasks:
                                if isinstance(task, dict):
                                    task_id = task.get('id', '')
                                    task['is_critical'] = task_id in critical_set
    
    def _optimize_resource_allocation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize resource allocation across tasks"""
        try:
            resources = plan.get('resources', [])
            tasks = self._extract_tasks(plan)
            
            if not resources or not tasks:
                return plan
            
            # Calculate resource utilization
            resource_utilization = {}
            resource_info = {}
            
            for resource in resources:
                if isinstance(resource, dict):
                    resource_id = resource.get('id') or resource.get('name', '')
                    if resource_id:
                        resource_utilization[resource_id] = 0.0
                        resource_info[resource_id] = resource
            
            # Analyze task resource assignments
            for task in tasks:
                if isinstance(task, dict):
                    assigned_resources = task.get('assigned_resources', [])
                    if not isinstance(assigned_resources, list):
                        continue
                    
                    task_duration = self._safe_float(task.get('duration'), 1.0)
                    
                    for resource_id in assigned_resources:
                        if isinstance(resource_id, str) and resource_id in resource_utilization:
                            resource_utilization[resource_id] += task_duration
            
            # Generate optimization data
            optimization_data = {
                'utilization': resource_utilization,
                'max_utilization': max(resource_utilization.values()) if resource_utilization else 0,
                'avg_utilization': (sum(resource_utilization.values()) / len(resource_utilization)) if resource_utilization else 0,
                'optimization_suggestions': self._generate_resource_suggestions(resource_utilization, resource_info)
            }
            
            plan['resource_optimization'] = optimization_data
            
        except Exception as e:
            logger.error(f"Resource optimization error: {e}")
        
        return plan
    
    def _generate_resource_suggestions(self, utilization: Dict[str, float], resource_info: Dict[str, Any]) -> List[str]:
        """Generate resource optimization suggestions"""
        suggestions = []
        
        if not utilization:
            return suggestions
        
        try:
            values = list(utilization.values())
            if not values:
                return suggestions
            
            max_util = max(values)
            min_util = min(values)
            avg_util = sum(values) / len(values)
            
            # High utilization warnings
            for resource_id, util in utilization.items():
                if util > avg_util * 1.5 and util > 10:  # Reasonable threshold
                    suggestions.append(f"Resource '{resource_id}' is highly utilized ({util:.1f} units). Consider additional capacity.")
            
            # Low utilization warnings
            for resource_id, util in utilization.items():
                if util > 0 and util < avg_util * 0.5 and avg_util > 2:  # Avoid noise for small projects
                    suggestions.append(f"Resource '{resource_id}' is underutilized ({util:.1f} units). Consider redistribution.")
            
            # Load balancing suggestion
            if max_util > 0 and min_util >= 0 and max_util > min_util * 2 and len(utilization) > 1:
                suggestions.append("Consider load balancing between resources to optimize utilization.")
            
            # Capacity warnings
            for resource_id, util in utilization.items():
                resource = resource_info.get(resource_id, {})
                availability = resource.get('availability', 1.0)
                
                availability = self._safe_float(availability, 1.0)
                if availability < 1.0 and util > 0:
                    effective_capacity = util / availability if availability > 0 else util
                    if effective_capacity > util * 1.2:
                        suggestions.append(f"Resource '{resource_id}' may be over-allocated considering {availability*100:.0f}% availability.")
        
        except Exception as e:
            logger.error(f"Error generating resource suggestions: {e}")
        
        return suggestions
    
    def _calculate_project_metrics(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive project metrics"""
        try:
            tasks = self._extract_tasks(plan)
            risks = plan.get('risks', [])
            
            # Validate risks data
            if not isinstance(risks, list):
                risks = []
            risks = [r for r in risks if isinstance(r, dict)]
            
            # Calculate basic metrics - Handle NoneType values safely
            total_duration = 0.0
            total_cost = 0.0
            total_effort = 0.0
            
            for task in tasks:
                # Safe duration calculation
                duration = task.get('duration', 0)
                try:
                    total_duration += float(duration) if duration is not None else 0.0
                except (ValueError, TypeError):
                    total_duration += 0.0
                
                # Safe cost calculation
                cost = task.get('cost')
                if cost is not None:
                    try:
                        total_cost += float(cost)
                    except (ValueError, TypeError):
                        pass  # Skip invalid cost values
                
                # Safe effort calculation
                effort = task.get('effort')
                if effort is not None:
                    try:
                        total_effort += float(effort)
                    except (ValueError, TypeError):
                        # Fallback to duration if effort is invalid
                        try:
                            total_effort += float(duration) if duration is not None else 0.0
                        except (ValueError, TypeError):
                            total_effort += 0.0
                else:
                    # Use duration as fallback for effort
                    try:
                        total_effort += float(duration) if duration is not None else 0.0
                    except (ValueError, TypeError):
                        total_effort += 0.0
            
            metrics = {
                'task_count': len(tasks),
                'total_estimated_effort': round(total_effort, 2),
                'total_estimated_cost': round(total_cost, 2),
                'total_estimated_duration': round(total_duration, 2),
                'average_task_duration': round(total_duration / len(tasks), 2) if tasks else 0,
                'risk_count': len(risks),
                'high_risk_count': len([r for r in risks if self._safe_float(r.get('risk_score'), 0) >= 15]),
                'total_risk_exposure': sum(self._safe_float(r.get('risk_score'), 0) for r in risks)
            }
            
            # Task complexity analysis
            complexity_score = self._calculate_complexity_score(tasks, plan.get('dependencies', []))
            metrics.update({
                'complexity_score': complexity_score,
                'complexity_level': self._get_complexity_level(complexity_score)
            })
            
            # Update project overview with calculated metrics
            overview = plan.get('project_overview', {})
            overview.update({
                'total_duration': metrics['total_estimated_duration'],
                'total_cost': metrics['total_estimated_cost'],
                'total_effort': metrics['total_estimated_effort']
            })
            plan['project_overview'] = overview
            
            plan['project_metrics'] = metrics
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
        
        return plan
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert a value to float, handling None and invalid types"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _calculate_complexity_score(self, tasks: List[Dict[str, Any]], dependencies: List[Dict[str, Any]]) -> float:
        """Calculate project complexity score"""
        if not tasks:
            return 0.0
        
        try:
            # Base complexity from task count
            task_complexity = len(tasks) * 0.1
            
            # Dependency complexity
            dep_count = len(dependencies) if isinstance(dependencies, list) else 0
            dependency_complexity = dep_count * 0.2
            
            # Duration variance complexity
            durations = []
            for task in tasks:
                duration = self._safe_float(task.get('duration'), 1.0)
                durations.append(duration)
            
            variance_complexity = 0.0
            if len(durations) > 1:
                avg_duration = sum(durations) / len(durations)
                variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
                variance_complexity = math.sqrt(variance) * 0.1
            
            # Resource complexity
            all_resources = set()
            for task in tasks:
                assigned = task.get('assigned_resources', [])
                if isinstance(assigned, list):
                    all_resources.update(str(r) for r in assigned if r)
            
            resource_complexity = len(all_resources) * 0.05
            
            total_complexity = task_complexity + dependency_complexity + variance_complexity + resource_complexity
            
            return round(total_complexity, 2)
        
        except Exception as e:
            logger.error(f"Complexity calculation error: {e}")
            return 1.0
    
    def _get_complexity_level(self, score: float) -> str:
        """Get complexity level description"""
        if score < 2.0:
            return "Low"
        elif score < 5.0:
            return "Medium"
        elif score < 10.0:
            return "High"
        else:
            return "Very High"
    
    def generate_schedule_recommendations(self, plan: Dict[str, Any]) -> List[str]:
        """Generate schedule optimization recommendations"""
        recommendations = []
        
        try:
            critical_path = plan.get('critical_path', [])
            if isinstance(critical_path, list) and critical_path:
                recommendations.append(f"Focus on critical path tasks: {', '.join(critical_path[:3])}...")
            
            metrics = plan.get('project_metrics', {})
            if isinstance(metrics, dict):
                complexity_level = metrics.get('complexity_level')
                if complexity_level == 'Very High':
                    recommendations.append("Consider breaking down complex tasks into smaller, manageable units.")
                
                high_risk_count = metrics.get('high_risk_count', 0)
                if high_risk_count > 3:
                    recommendations.append("High number of high-risk items. Prioritize risk mitigation activities.")
                
                avg_duration = metrics.get('average_task_duration', 0)
                if avg_duration > 5:
                    recommendations.append("Tasks have long average duration. Consider more granular breakdown.")
            
            resource_optimization = plan.get('resource_optimization', {})
            if isinstance(resource_optimization, dict):
                suggestions = resource_optimization.get('optimization_suggestions', [])
                if isinstance(suggestions, list):
                    recommendations.extend(suggestions)
        
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
        
        return recommendations