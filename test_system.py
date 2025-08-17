#!/usr/bin/env python3
"""
Test script for NeuroAlign system components
"""

import sys
import os
import asyncio

# Add the neuroalign directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'neuroalign'))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from neuroalign.utils.config import settings
        print("✅ Config imported successfully")
        
        from neuroalign.models.database import User, FatigueRecord, BioRhythmRecord, Schedule
        print("✅ Database models imported successfully")
        
        from neuroalign.services.fatigue_detector import FatigueDetector
        print("✅ Fatigue detector imported successfully")
        
        from neuroalign.services.bio_rhythm_analyzer import BioRhythmAnalyzer
        print("✅ Bio-rhythm analyzer imported successfully")
        
        from neuroalign.services.websocket_manager import WebSocketManager
        print("✅ WebSocket manager imported successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

async def test_fatigue_detector():
    """Test fatigue detector functionality"""
    print("\n🧪 Testing fatigue detector...")
    
    try:
        detector = FatigueDetector()
        print("✅ Fatigue detector initialized")
        
        # Test with dummy data
        typing_data = {
            "keystrokes": [
                {"type": "keypress", "timestamp": "2023-01-01T10:00:00"},
                {"type": "keypress", "timestamp": "2023-01-01T10:00:01"}
            ],
            "backspaces": [],
            "hesitations": []
        }
        
        result = await detector.analyze_typing(typing_data)
        print(f"✅ Typing analysis result: {result['typing_fatigue_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fatigue detector error: {e}")
        return False

async def test_bio_rhythm_analyzer():
    """Test bio-rhythm analyzer functionality"""
    print("\n🧪 Testing bio-rhythm analyzer...")
    
    try:
        analyzer = BioRhythmAnalyzer()
        print("✅ Bio-rhythm analyzer initialized")
        
        # Test with dummy data
        bio_signals = {
            "heart_rate": 75.0,
            "sleep_duration": 8.0,
            "sleep_quality": 0.8,
            "steps_count": 8000,
            "stress_level": 0.3
        }
        
        result = await analyzer.predict_energy(bio_signals)
        print(f"✅ Energy prediction: {result['current_energy_level']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Bio-rhythm analyzer error: {e}")
        return False

def test_websocket_manager():
    """Test WebSocket manager functionality"""
    print("\n🧪 Testing WebSocket manager...")
    
    try:
        manager = WebSocketManager()
        print("✅ WebSocket manager initialized")
        
        stats = manager.get_connection_stats()
        print(f"✅ Connection stats: {stats['total_connections']} connections")
        
        return True
        
    except Exception as e:
        print(f"❌ WebSocket manager error: {e}")
        return False

async def main():
    """Run all tests"""
    print("🧠 NeuroAlign System Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Exiting.")
        return False
    
    # Test components
    fatigue_ok = await test_fatigue_detector()
    bio_ok = await test_bio_rhythm_analyzer()
    ws_ok = test_websocket_manager()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Fatigue Detector: {'✅ PASS' if fatigue_ok else '❌ FAIL'}")
    print(f"   Bio-Rhythm Analyzer: {'✅ PASS' if bio_ok else '❌ FAIL'}")
    print(f"   WebSocket Manager: {'✅ PASS' if ws_ok else '❌ FAIL'}")
    
    all_passed = fatigue_ok and bio_ok and ws_ok
    
    if all_passed:
        print("\n🎉 All tests passed! System is ready to run.")
        print("   Run 'python run.py' to start the application.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())