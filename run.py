#!/usr/bin/env python3
"""
NeuroAlign - AI-Powered Fatigue Detection & Smart Scheduling System
Startup script
"""

import uvicorn
import os
import sys

# Add the neuroalign directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neuroalign'))

if __name__ == "__main__":
    print("🧠 Starting NeuroAlign Fatigue Detection System...")
    print("📊 Dashboard: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔌 WebSocket: ws://localhost:8000/ws/fatigue")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "neuroalign.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🛑 Shutting down NeuroAlign system...")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)