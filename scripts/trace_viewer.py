#!/usr/bin/env python3
"""
Mimir Trace Viewer - Real-time monitoring of Mimir's deep research capabilities

This script connects to NATS JetStream and displays real-time traces from
the monitored Mimir server, providing insights into:
- Tool execution patterns
- Research session flows  
- Query complexity analysis
- Performance metrics
- Error tracking

Usage:
    python scripts/trace_viewer.py [--nats-url nats://localhost:4222]
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

try:
    import nats
    from nats.js import JetStreamContext
    HAS_NATS = True
except ImportError:
    HAS_NATS = False
    print("NATS not available. Install with: pip install nats-py")
    sys.exit(1)

# Color formatting for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class MimirTraceViewer:
    """Real-time trace viewer for Mimir monitoring data."""
    
    def __init__(self, nats_url: str = "nats://localhost:4222", stream_name: str = "MIMIR_TRACES"):
        self.nats_url = nats_url
        self.stream_name = stream_name
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[JetStreamContext] = None
        self.running = True
        
        # Statistics tracking
        self.stats = {
            "total_traces": 0,
            "tool_calls": {},
            "sessions": {},
            "errors": 0,
            "avg_execution_time": 0.0
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    async def connect(self):
        """Connect to NATS JetStream."""
        try:
            self.nc = await nats.connect(self.nats_url)
            self.js = self.nc.jetstream()
            
            print(f"{Colors.OKGREEN}✓ Connected to NATS at {self.nats_url}{Colors.ENDC}")
            print(f"{Colors.OKBLUE}📊 Monitoring stream: {self.stream_name}{Colors.ENDC}")
            print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to connect to NATS: {e}{Colors.ENDC}")
            return False
    
    async def start_monitoring(self):
        """Start monitoring traces from all subjects."""
        if not self.js:
            return
        
        try:
            # Subscribe to all trace subjects
            subject_pattern = f"{self.stream_name.lower()}.*"
            
            async def message_handler(msg):
                await self.handle_trace_message(msg)
            
            # Create pull subscription
            sub = await self.js.pull_subscribe(
                subject_pattern,
                "trace_viewer_group",
                config=nats.js.api.ConsumerConfig(
                    deliver_policy=nats.js.api.DeliverPolicy.ALL,
                    ack_policy=nats.js.api.AckPolicy.EXPLICIT
                )
            )
            
            print(f"{Colors.OKCYAN}🎯 Subscribed to {subject_pattern}{Colors.ENDC}")
            print(f"{Colors.OKGREEN}🚀 Trace viewer started - press Ctrl+C to stop{Colors.ENDC}")
            print()
            
            # Poll for messages
            while self.running:
                try:
                    messages = await sub.fetch(1, timeout=1.0)
                    for msg in messages:
                        await message_handler(msg)
                        await msg.ack()
                        
                except nats.errors.TimeoutError:
                    # No messages, continue polling
                    continue
                except Exception as e:
                    self.logger.error(f"Error processing messages: {e}")
                    
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error starting monitoring: {e}{Colors.ENDC}")
    
    async def handle_trace_message(self, msg):
        """Handle incoming trace messages."""
        try:
            # Parse the trace data
            trace_data = json.loads(msg.data.decode())
            
            # Update statistics
            self.stats["total_traces"] += 1
            
            # Display the trace
            await self.display_trace(trace_data)
            
        except Exception as e:
            self.logger.error(f"Error handling trace message: {e}")
    
    async def display_trace(self, trace_data: Dict[str, Any]):
        """Display trace data in a formatted way."""
        trace_type = trace_data.get("trace_type", "unknown")
        tool_name = trace_data.get("tool_name", "unknown")
        timestamp = trace_data.get("timestamp", 0)
        
        # Format timestamp
        dt = datetime.fromtimestamp(timestamp)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # Milliseconds
        
        # Color code by trace type
        if "start" in trace_type:
            color = Colors.OKBLUE
            icon = "🚀"
        elif "complete" in trace_type:
            color = Colors.OKGREEN
            icon = "✅"
        elif "error" in trace_type:
            color = Colors.FAIL
            icon = "❌"
            self.stats["errors"] += 1
        else:
            color = Colors.OKCYAN
            icon = "📊"
        
        # Display basic trace info
        print(f"{color}{icon} {time_str} {tool_name.upper():<20} {trace_type}{Colors.ENDC}")
        
        # Display specific details based on operation
        operation = trace_data.get("operation")
        if operation == "repo_indexing":
            self.display_indexing_trace(trace_data)
        elif operation == "repo_search":
            self.display_search_trace(trace_data)
        elif operation == "deep_research":
            self.display_research_trace(trace_data)
        
        # Display performance metrics
        if "execution_time" in trace_data:
            exec_time = trace_data["execution_time"]
            if exec_time > 5.0:
                perf_color = Colors.WARNING
            elif exec_time > 10.0:
                perf_color = Colors.FAIL
            else:
                perf_color = Colors.OKGREEN
            
            print(f"  {perf_color}⏱️  Execution time: {exec_time:.3f}s{Colors.ENDC}")
        
        # Display errors
        if trace_data.get("error"):
            print(f"  {Colors.FAIL}💥 Error: {trace_data['error']}{Colors.ENDC}")
        
        print()  # Blank line for readability
    
    def display_indexing_trace(self, trace_data: Dict[str, Any]):
        """Display repository indexing specific information."""
        repo_path = trace_data.get("repo_path", "unknown")
        language = trace_data.get("language", "unknown")
        
        print(f"  📁 Repository: {repo_path}")
        print(f"  🔤 Language: {language}")
        
        if "index_id" in trace_data:
            index_id = trace_data["index_id"]
            print(f"  🆔 Index ID: {index_id[:12]}...")
            
            # Track session
            if index_id not in self.stats["sessions"]:
                self.stats["sessions"][index_id] = {
                    "created_at": trace_data.get("timestamp", 0),
                    "repo_path": repo_path,
                    "operations": 0
                }
    
    def display_search_trace(self, trace_data: Dict[str, Any]):
        """Display search operation specific information."""
        query = trace_data.get("query", "")
        k = trace_data.get("k", 0)
        results_count = trace_data.get("results_count", 0)
        
        print(f"  🔍 Query: '{query[:50]}...' ({len(query)} chars)")
        print(f"  📊 Results: {results_count}/{k} requested")
        
        features = trace_data.get("features", {})
        if features:
            enabled_features = [f for f, enabled in features.items() if enabled]
            print(f"  🎛️  Features: {', '.join(enabled_features)}")
    
    def display_research_trace(self, trace_data: Dict[str, Any]):
        """Display deep research specific information."""
        question = trace_data.get("question", "")
        complexity = trace_data.get("question_complexity", "unknown")
        evidence_count = trace_data.get("evidence_count", 0)
        
        # Color code complexity
        complexity_colors = {
            "low": Colors.OKGREEN,
            "medium": Colors.WARNING,
            "high": Colors.FAIL
        }
        complexity_color = complexity_colors.get(complexity, Colors.OKCYAN)
        
        print(f"  🤔 Question: '{question[:80]}...' ({len(question)} chars)")
        print(f"  {complexity_color}🧠 Complexity: {complexity.upper()}{Colors.ENDC}")
        
        if evidence_count > 0:
            print(f"  📚 Evidence: {evidence_count} sources")
        
        answer_length = trace_data.get("answer_length", 0)
        if answer_length > 0:
            confidence = trace_data.get("answer_confidence", "unknown")
            print(f"  📝 Answer: {answer_length} chars, confidence: {confidence}")
    
    def display_stats(self):
        """Display accumulated statistics."""
        print(f"\n{Colors.HEADER}📈 MONITORING STATISTICS{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Total traces: {self.stats['total_traces']}{Colors.ENDC}")
        print(f"{Colors.OKBLUE}Active sessions: {len(self.stats['sessions'])}{Colors.ENDC}")
        print(f"{Colors.FAIL if self.stats['errors'] > 0 else Colors.OKGREEN}Errors: {self.stats['errors']}{Colors.ENDC}")
        
        if self.stats["tool_calls"]:
            print(f"\n{Colors.OKCYAN}Tool usage:{Colors.ENDC}")
            for tool, count in self.stats["tool_calls"].items():
                print(f"  {tool}: {count}")
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\n{Colors.WARNING}📊 Shutting down trace viewer...{Colors.ENDC}")
        self.display_stats()
        self.running = False
    
    async def run(self):
        """Main run loop."""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Connect to NATS
        if not await self.connect():
            return
        
        try:
            # Start monitoring
            await self.start_monitoring()
            
        finally:
            # Clean up
            if self.nc:
                await self.nc.close()
                print(f"{Colors.OKGREEN}✓ Disconnected from NATS{Colors.ENDC}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mimir Trace Viewer - Real-time monitoring dashboard"
    )
    parser.add_argument(
        "--nats-url",
        default="nats://localhost:4222",
        help="NATS server URL"
    )
    parser.add_argument(
        "--stream-name",
        default="MIMIR_TRACES",
        help="NATS stream name"
    )
    
    args = parser.parse_args()
    
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("🔍 Mimir Deep Research Trace Viewer")
    print("Real-time monitoring of AI-powered code research")
    print(f"{'='*50}{Colors.ENDC}")
    
    viewer = MimirTraceViewer(args.nats_url, args.stream_name)
    await viewer.run()


if __name__ == "__main__":
    if not HAS_NATS:
        print("NATS Python client not available.")
        print("Install with: pip install nats-py")
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Handled by signal handler