"""
Real-time Updates System for PlannerIA
Système de mises à jour temps réel avec streaming, WebSocket et monitoring
"""

from .streaming_engine import (
    RealTimeStreamingEngine,
    StreamEvent,
    StreamEventType,
    StreamStatus,
    AgentStatus,
    get_streaming_engine,
    start_realtime_session,
    emit_agent_progress,
    emit_plan_update
)

from .live_visualization import (
    LivePlanVisualization,
    get_live_visualization,
    render_live_generation_dashboard,
    render_live_plan_preview
)

from .progress_tracker import (
    AdvancedProgressTracker,
    ProgressIndicator,
    ProgressStep,
    ProgressType,
    StatusLevel,
    get_progress_tracker,
    render_progress_dashboard,
    update_agent_progress,
    create_progress_indicator
)

from .websocket_manager import (
    WebSocketManager,
    WebSocketClient,
    WebSocketMessage,
    WebSocketMessageType,
    WebSocketEventChannel,
    get_websocket_manager,
    start_websocket_server,
    run_websocket_server_in_thread,
    SimpleWebSocketClient
)

from .agent_monitor import (
    LiveAgentMonitor,
    AgentMetrics,
    AgentHealthStatus,
    SystemAlert,
    MonitoringMetric,
    get_agent_monitor,
    render_agent_monitoring_dashboard,
    get_agent_health_status
)

__all__ = [
    # Streaming Engine
    'RealTimeStreamingEngine',
    'StreamEvent', 
    'StreamEventType',
    'StreamStatus',
    'AgentStatus',
    'get_streaming_engine',
    'start_realtime_session',
    'emit_agent_progress',
    'emit_plan_update',
    
    # Live Visualization
    'LivePlanVisualization',
    'get_live_visualization',
    'render_live_generation_dashboard',
    'render_live_plan_preview',
    
    # Progress Tracking
    'AdvancedProgressTracker',
    'ProgressIndicator',
    'ProgressStep',
    'ProgressType',
    'StatusLevel',
    'get_progress_tracker',
    'render_progress_dashboard',
    'update_agent_progress',
    'create_progress_indicator',
    
    # WebSocket Communication
    'WebSocketManager',
    'WebSocketClient',
    'WebSocketMessage',
    'WebSocketMessageType',
    'WebSocketEventChannel',
    'get_websocket_manager',
    'start_websocket_server',
    'run_websocket_server_in_thread',
    'SimpleWebSocketClient',
    
    # Agent Monitoring
    'LiveAgentMonitor',
    'AgentMetrics',
    'AgentHealthStatus',
    'SystemAlert',
    'MonitoringMetric',
    'get_agent_monitor',
    'render_agent_monitoring_dashboard',
    'get_agent_health_status'
]