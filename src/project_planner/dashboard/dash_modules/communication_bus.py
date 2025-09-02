"""
🚀 PlannerIA Communication Bus - Universal Data Exchange System
Real-time data synchronization between all modules with event-driven architecture
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events in the communication bus"""
    DATA_UPDATE = "data_update"
    USER_ACTION = "user_action"
    AI_PREDICTION = "ai_prediction"
    ALERT_TRIGGERED = "alert_triggered"
    MODULE_READY = "module_ready"
    EXPORT_REQUEST = "export_request"
    SYNC_REQUEST = "sync_request"

class Priority(Enum):
    """Event priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class BusEvent:
    """Standardized event structure for the communication bus"""
    event_id: str
    event_type: EventType
    source_module: str
    target_modules: List[str]  # ["*"] for broadcast
    payload: Dict[str, Any]
    timestamp: str
    priority: Priority = Priority.MEDIUM
    processed: bool = False
    metadata: Optional[Dict[str, Any]] = None

class CommunicationBus:
    """
    Universal communication system for all PlannerIA modules
    Enables real-time data exchange, event propagation, and state synchronization
    """
    
    def __init__(self):
        self.events_queue: List[BusEvent] = []
        self.subscribers: Dict[str, List[Callable]] = {}
        self.module_states: Dict[str, Dict[str, Any]] = {}
        self.global_context: Dict[str, Any] = {
            'project_data': {},
            'ai_status': {'active': False, 'models_loaded': False},
            'user_preferences': {},
            'session_info': {
                'started': datetime.now().isoformat(),
                'last_activity': datetime.now().isoformat()
            }
        }
        
        logger.info("Communication Bus initialized")
    
    def publish_event(self, event: BusEvent) -> bool:
        """Publishes an event to the bus"""
        try:
            # Add timestamp if not provided
            if not event.timestamp:
                event.timestamp = datetime.now().isoformat()
            
            # Add to queue
            self.events_queue.append(event)
            
            # Sort by priority (highest first)
            self.events_queue.sort(key=lambda e: e.priority.value, reverse=True)
            
            # Notify subscribers immediately for high priority events
            if event.priority in [Priority.HIGH, Priority.CRITICAL]:
                self._notify_subscribers(event)
            
            logger.debug(f"Event published: {event.event_type} from {event.source_module}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    def subscribe(self, module_name: str, callback: Callable) -> bool:
        """Subscribes a module to bus events"""
        try:
            if module_name not in self.subscribers:
                self.subscribers[module_name] = []
            
            self.subscribers[module_name].append(callback)
            logger.info(f"Module {module_name} subscribed to communication bus")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe {module_name}: {e}")
            return False
    
    def update_module_state(self, module_name: str, state: Dict[str, Any]) -> bool:
        """Updates the state of a specific module"""
        try:
            self.module_states[module_name] = {
                **state,
                'last_updated': datetime.now().isoformat()
            }
            
            # Publish state update event
            event = BusEvent(
                event_id=f"state_update_{int(time.time() * 1000)}",
                event_type=EventType.DATA_UPDATE,
                source_module=module_name,
                target_modules=["*"],  # Broadcast to all
                payload={'new_state': state},
                timestamp=datetime.now().isoformat()
            )
            
            self.publish_event(event)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update state for {module_name}: {e}")
            return False
    
    def get_module_state(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Gets the current state of a module"""
        return self.module_states.get(module_name)
    
    def get_global_context(self) -> Dict[str, Any]:
        """Gets the global application context"""
        return self.global_context.copy()
    
    def update_global_context(self, updates: Dict[str, Any]) -> bool:
        """Updates the global application context"""
        try:
            self.global_context.update(updates)
            self.global_context['session_info']['last_activity'] = datetime.now().isoformat()
            
            # Publish global context update
            event = BusEvent(
                event_id=f"global_update_{int(time.time() * 1000)}",
                event_type=EventType.DATA_UPDATE,
                source_module="system",
                target_modules=["*"],
                payload={'context_updates': updates},
                timestamp=datetime.now().isoformat(),
                priority=Priority.HIGH
            )
            
            self.publish_event(event)
            return True
            
        except Exception as e:
            logger.error(f"Failed to update global context: {e}")
            return False
    
    def process_events(self) -> List[BusEvent]:
        """Processes all pending events and returns processed events"""
        processed_events = []
        
        for event in self.events_queue[:]:
            if not event.processed:
                try:
                    # Notify all relevant subscribers
                    self._notify_subscribers(event)
                    event.processed = True
                    processed_events.append(event)
                    
                except Exception as e:
                    logger.error(f"Failed to process event {event.event_id}: {e}")
        
        # Clean up processed events (keep last 100 for history)
        self.events_queue = [e for e in self.events_queue if not e.processed][-100:]
        
        return processed_events
    
    def _notify_subscribers(self, event: BusEvent):
        """Notifies relevant subscribers about an event"""
        target_modules = event.target_modules
        
        for module_name, callbacks in self.subscribers.items():
            # Check if this module should receive the event
            should_notify = (
                "*" in target_modules or 
                module_name in target_modules or
                module_name == event.source_module
            )
            
            if should_notify:
                for callback in callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Callback failed for {module_name}: {e}")
    
    def create_quick_action_event(self, action_type: str, source_module: str, 
                                 data: Dict[str, Any]) -> BusEvent:
        """Creates a standardized quick action event"""
        return BusEvent(
            event_id=f"quick_action_{int(time.time() * 1000)}",
            event_type=EventType.USER_ACTION,
            source_module=source_module,
            target_modules=["project_manager", "data_persistence"],
            payload={
                'action_type': action_type,
                'data': data,
                'requires_confirmation': action_type in ['delete', 'reset', 'bulk_update']
            },
            timestamp=datetime.now().isoformat(),
            priority=Priority.HIGH if action_type in ['delete', 'reset'] else Priority.MEDIUM
        )
    
    def get_module_health(self) -> Dict[str, Any]:
        """Returns health status of all modules"""
        health_status = {}
        
        for module_name, state in self.module_states.items():
            last_update = state.get('last_updated')
            if last_update:
                # Check if module is responsive (updated within last 30 seconds)
                try:
                    last_time = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                    time_diff = (datetime.now() - last_time.replace(tzinfo=None)).total_seconds()
                    
                    health_status[module_name] = {
                        'status': 'healthy' if time_diff < 30 else 'stale',
                        'last_seen': last_update,
                        'response_time': time_diff
                    }
                except:
                    health_status[module_name] = {'status': 'error', 'last_seen': last_update}
            else:
                health_status[module_name] = {'status': 'unknown', 'last_seen': None}
        
        return health_status
    
    def export_bus_data(self, format: str = 'json') -> str:
        """Exports all bus data for communication/sharing"""
        export_data = {
            'global_context': self.global_context,
            'module_states': self.module_states,
            'recent_events': [asdict(e) for e in self.events_queue[-50:]],  # Last 50 events
            'health_status': self.get_module_health(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        else:
            return str(export_data)

# === SINGLETON INSTANCE ===
_bus_instance = None

def get_communication_bus() -> CommunicationBus:
    """Gets the singleton communication bus instance"""
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = CommunicationBus()
    return _bus_instance

# === CONVENIENCE FUNCTIONS FOR MODULES ===

def quick_publish(event_type: EventType, source: str, target: List[str], data: Dict[str, Any]):
    """Quick way to publish an event"""
    bus = get_communication_bus()
    event = BusEvent(
        event_id=f"{source}_{int(time.time() * 1000)}",
        event_type=event_type,
        source_module=source,
        target_modules=target,
        payload=data,
        timestamp=datetime.now().isoformat()
    )
    return bus.publish_event(event)

def quick_subscribe(module_name: str, callback: Callable):
    """Quick way to subscribe to bus events"""
    bus = get_communication_bus()
    return bus.subscribe(module_name, callback)

def update_state(module_name: str, state: Dict[str, Any]):
    """Quick way to update module state"""
    bus = get_communication_bus()
    return bus.update_module_state(module_name, state)

def get_context():
    """Quick way to get global context"""
    bus = get_communication_bus()
    return bus.get_global_context()

def apply_to_project(source_module: str, changes: Dict[str, Any]):
    """Standard 'Apply to Project' quick action"""
    return quick_publish(
        EventType.USER_ACTION,
        source_module,
        ["project_manager"],
        {
            'action': 'apply_changes',
            'changes': changes,
            'timestamp': datetime.now().isoformat()
        }
    )

def export_data(source_module: str, data: Dict[str, Any], format: str = 'json'):
    """Standard 'Export' quick action"""
    return quick_publish(
        EventType.EXPORT_REQUEST,
        source_module,
        ["export_manager"],
        {
            'action': 'export_data',
            'data': data,
            'format': format,
            'timestamp': datetime.now().isoformat()
        }
    )

def share_insights(source_module: str, insights: Dict[str, Any]):
    """Standard 'Share' quick action"""
    return quick_publish(
        EventType.USER_ACTION,
        source_module,
        ["collaboration_manager"],
        {
            'action': 'share_insights',
            'insights': insights,
            'timestamp': datetime.now().isoformat()
        }
    )