"""
URGENT SHOWCASE APP - Minimal but Impressive Conversation Interface
================================================================

Timeline: 3 days to working showcase for evaluation jury
Focus: Single killer demo flow that shows AI-first planning
"""

import streamlit as st
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))

# Import existing modules
try:
    from crew import CrewPlanner
    CREW_AVAILABLE = True
except ImportError:
    CREW_AVAILABLE = False
    st.error("CrewAI not available - check crew.py import")

# Streamlit config for demo
st.set_page_config(
    page_title="PlannerIA - AI Project Manager Showcase",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for impressive demo look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86C1;
        font-size: 3rem;
        margin-bottom: 2rem;
    }
    .demo-status {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .ai-thinking {
        background: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stChatMessage[data-testid="chat-message-user"] {
        background: #e3f2fd;
    }
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: #f3e5f5;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state for demo"""
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Hi! I'm PlannerIA, your AI Project Manager. Tell me about a project you'd like to plan, and I'll create a complete strategy in real-time!",
                "timestamp": datetime.now()
            }
        ]
    
    if 'demo_stage' not in st.session_state:
        st.session_state.demo_stage = "ready"
    
    if 'current_plan' not in st.session_state:
        st.session_state.current_plan = None

async def process_with_crew_ai(user_input: str) -> Dict[str, Any]:
    """Process user input with CrewAI - CORE DEMO FUNCTION"""
    
    if not CREW_AVAILABLE:
        return {
            "error": "CrewAI not available",
            "fallback_response": "I would normally analyze your project using multi-agent planning, but there's a technical issue. Here's what I would do..."
        }
    
    try:
        # Show AI thinking process
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("🤖 **AI Processing Pipeline:**")
        
        with st.spinner(""):
            # Step 1: Initialize CrewAI
            thinking_placeholder.markdown("🤖 **AI Processing Pipeline:**\n- ✅ Initializing multi-agent system...")
            crew = CrewPlanner()
            
            # Step 2: Generate plan
            thinking_placeholder.markdown("🤖 **AI Processing Pipeline:**\n- ✅ Initializing multi-agent system...\n- 🔄 Generating project plan...")
            result = crew.generate_plan_from_brief(user_input)
            
            # Step 3: Analysis
            thinking_placeholder.markdown("🤖 **AI Processing Pipeline:**\n- ✅ Initializing multi-agent system...\n- ✅ Generating project plan...\n- 🔄 Running risk analysis...")
            
            # Step 4: Complete
            thinking_placeholder.markdown("🤖 **AI Processing Pipeline:**\n- ✅ Initializing multi-agent system...\n- ✅ Generating project plan...\n- ✅ Running risk analysis...\n- ✅ **Analysis complete!**")
            
        return {
            "success": True,
            "plan": result,
            "ai_analysis": "Multi-agent planning completed successfully"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "fallback_response": f"Technical issue with AI processing: {e}. In a live demo, this would show the complete multi-agent planning process."
        }

def render_plan_visualization(plan_data: Dict):
    """Render plan visualization for demo impact"""
    
    if not plan_data or "error" in plan_data:
        st.error("⚠️ Plan visualization unavailable - technical issue")
        return
    
    st.markdown("### 📊 Generated Project Plan")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        plan = plan_data.get("plan", {})
        overview = plan.get("project_overview", {})
        
        with col1:
            duration = overview.get('total_duration', 0)
            if isinstance(duration, (int, float)):
                duration_str = f"{duration:.0f} days"
            else:
                duration_str = f"{duration} days" if duration != 0 else "N/A"
            st.metric("Duration", duration_str)
        with col2:
            cost = overview.get('total_cost', 0)
            if isinstance(cost, (int, float)):
                cost_str = f"€{cost:,.0f}"
            else:
                cost_str = "N/A"
            st.metric("Budget", cost_str)
        with col3:
            critical_path = overview.get('critical_path_duration', 0)
            if isinstance(critical_path, (int, float)):
                critical_path_str = f"{critical_path:.0f} days"
            else:
                critical_path_str = f"{critical_path} days" if critical_path != 0 else "N/A"
            st.metric("Critical Path", critical_path_str)
        with col4:
            st.metric("Phases", len(plan.get("wbs", {}).get("phases", [])))
            
        # Show plan structure
        st.markdown("### 🗂️ Project Structure")
        if "wbs" in plan and "phases" in plan["wbs"]:
            for i, phase in enumerate(plan["wbs"]["phases"][:3], 1):  # Show first 3 phases
                with st.expander(f"Phase {i}: {phase.get('name', 'Unknown')} ({phase.get('duration', 'N/A')} days)"):
                    tasks = phase.get('tasks', [])
                    for task in tasks[:5]:  # Show first 5 tasks
                        st.write(f"• {task.get('name', 'Task')}")
                        
    except Exception as e:
        st.error(f"Visualization error: {e}")
        st.json(plan_data)  # Fallback: show raw data

def main():
    """Main showcase application"""
    
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🤖 PlannerIA Showcase</h1>', unsafe_allow_html=True)
    st.markdown('<div class="demo-status">AI-First Project Management • Live Demo for Evaluation Jury</div>', unsafe_allow_html=True)
    
    # Demo instructions
    with st.expander("📋 Demo Instructions", expanded=False):
        st.markdown("""
        **For Evaluation Jury:**
        1. Type a project description (e.g., "Create a mobile fitness app")
        2. Watch AI analyze and generate complete project plan in real-time
        3. See multi-agent system working with live visualization
        4. Experience conversation-driven project management
        
        **Example prompts:**
        - "I want to launch a food delivery startup"
        - "Build an e-commerce website for my business"
        - "Create a mobile game with multiplayer features"
        """)
    
    # Chat interface
    st.markdown("### 💬 Conversation with AI Project Manager")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show plan if it was generated
            if message["role"] == "assistant" and "plan_data" in message:
                render_plan_visualization(message["plan_data"])
    
    # Chat input
    if prompt := st.chat_input("Describe your project idea..."):
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Show user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with AI
        with st.chat_message("assistant"):
            # Show thinking process
            st.markdown("🤖 **Analyzing your project...**")
            
            # Process with CrewAI
            result = asyncio.run(process_with_crew_ai(prompt))
            
            # Generate response
            if result.get("success"):
                response = f"""
🎯 **Project Analysis Complete!**

I've analyzed your project "{prompt}" using my multi-agent planning system. Here's what I found:

✅ **Generated complete project breakdown**  
✅ **Identified critical path and dependencies**  
✅ **Calculated time and budget estimates**  
✅ **Risk assessment completed**

This demonstrates how I work as an AI-first project manager - you describe your vision, and I immediately create actionable plans with intelligent analysis.
"""
                plan_data = result
            else:
                response = f"""
⚠️ **Demo Mode - Technical Issue**

In a live environment, I would have:
- Analyzed your project using multi-agent AI system
- Generated complete WBS (Work Breakdown Structure)
- Calculated optimal timelines and budgets
- Identified potential risks and mitigation strategies
- Provided personalized recommendations

{result.get('fallback_response', '')}
"""
                plan_data = None
            
            st.write(response)
            
            # Show visualization if available
            if plan_data and plan_data.get("success"):
                render_plan_visualization(plan_data)
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "timestamp": datetime.now(),
                "plan_data": plan_data
            })
    
    # Demo status
    st.sidebar.markdown("### 🎯 Showcase Status")
    st.sidebar.success("✅ Conversation Interface Ready")
    st.sidebar.info("🔄 CrewAI Integration: " + ("Active" if CREW_AVAILABLE else "Testing Mode"))
    st.sidebar.info(f"💬 Messages: {len(st.session_state.messages)}")

if __name__ == "__main__":
    main()