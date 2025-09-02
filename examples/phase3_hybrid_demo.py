"""
Phase 3 Hybrid Query Engine Demonstration.

Demonstrates the complete Phase 3 implementation with intelligent search synthesis,
multiple strategies, and performance optimization.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

# Import Phase 3 components
from repoindex.pipeline.hybrid_query_engine import (
    HybridQueryEngine,
    QueryContext,
    QueryStrategy,
    QueryType,
)
from repoindex.pipeline.advanced_query_processor import create_query_processor
from repoindex.pipeline.performance_optimizer import (
    create_performance_optimizer,
    create_standard_benchmark_suite,
)
from repoindex.data.schemas import (
    VectorIndex,
    VectorChunk,
    SerenaGraph,
    SerenaEntry,
    SerenaType,
    RepoMap,
    RepoEdge,
    FileRank,
)


class Phase3Demo:
    """Comprehensive demonstration of Phase 3 capabilities."""
    
    def __init__(self):
        self.hybrid_engine = HybridQueryEngine()
        self.query_processor = create_query_processor()
        self.performance_optimizer = None
        self.demo_search_data = self._create_demo_data()
    
    def _create_demo_data(self) -> Dict[str, Any]:
        """Create comprehensive demo data for testing."""
        # Create diverse vector chunks
        vector_chunks = [
            VectorChunk(
                path="src/auth/authentication.py",
                span=(15, 30),
                hash="auth123",
                text="""def authenticate_user(username: str, password: str) -> bool:
    '''Authenticate user with username and password'''
    if not username or not password:
        return False
    
    user = find_user_by_username(username)
    if not user:
        return False
    
    return verify_password(password, user.password_hash)""",
                embedding=[0.1, 0.3, 0.5, 0.2, 0.4]
            ),
            VectorChunk(
                path="src/auth/password_utils.py", 
                span=(8, 20),
                hash="pwd456",
                text="""import bcrypt

def hash_password(password: str) -> str:
    '''Hash password using bcrypt'''
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    '''Verify password against hash'''
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))""",
                embedding=[0.2, 0.4, 0.3, 0.5, 0.1]
            ),
            VectorChunk(
                path="src/api/user_controller.py",
                span=(25, 45),
                hash="ctrl789",
                text="""class UserController:
    '''Handle user-related API endpoints'''
    
    def __init__(self, auth_service: AuthService):
        self.auth_service = auth_service
    
    async def login(self, request: LoginRequest) -> LoginResponse:
        '''Login user and return JWT token'''
        if not await self.auth_service.authenticate(
            request.username, request.password
        ):
            raise HTTPException(401, "Invalid credentials")
        
        token = self.auth_service.generate_jwt_token(request.username)
        return LoginResponse(token=token, expires_in=3600)""",
                embedding=[0.3, 0.2, 0.4, 0.1, 0.5]
            ),
            VectorChunk(
                path="src/database/user_repository.py",
                span=(10, 25),
                hash="repo101",
                text="""class UserRepository:
    '''Database operations for user management'''
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    async def find_by_username(self, username: str) -> Optional[User]:
        '''Find user by username'''
        return await self.db.query(User).filter(
            User.username == username
        ).first()
    
    async def create_user(self, user_data: dict) -> User:
        '''Create new user in database'''
        user = User(**user_data)
        self.db.add(user)
        await self.db.commit()
        return user""",
                embedding=[0.4, 0.1, 0.2, 0.5, 0.3]
            ),
            VectorChunk(
                path="src/utils/jwt_helper.py",
                span=(5, 18),
                hash="jwt202",
                text="""import jwt
from datetime import datetime, timedelta

def generate_jwt_token(user_id: str, secret: str, expires_hours: int = 24) -> str:
    '''Generate JWT token for user'''
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(hours=expires_hours),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, secret, algorithm='HS256')

def verify_jwt_token(token: str, secret: str) -> dict:
    '''Verify and decode JWT token'''
    try:
        return jwt.decode(token, secret, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token")""",
                embedding=[0.5, 0.3, 0.1, 0.4, 0.2]
            ),
            VectorChunk(
                path="tests/test_authentication.py",
                span=(1, 15),
                hash="test303",
                text="""import pytest
from auth.authentication import authenticate_user
from auth.password_utils import hash_password

class TestAuthentication:
    def test_valid_authentication(self):
        '''Test successful user authentication'''
        # Setup test data
        password = "secure_password123"
        hashed = hash_password(password)
        
        # Mock user lookup
        with patch('src.auth.authentication.find_user_by_username') as mock_find:
            mock_find.return_value = MockUser(password_hash=hashed)
            
            result = authenticate_user("testuser", password)
            assert result is True""",
                embedding=[0.2, 0.5, 0.4, 0.3, 0.1]
            )
        ]
        
        # Create Serena graph entries
        serena_entries = [
            SerenaEntry(
                path="src/auth/authentication.py",
                span=(15, 30),
                type=SerenaType.DEF,
                symbol="authenticate_user",
                sig="def authenticate_user(username: str, password: str) -> bool",
                doc="Authenticate user with username and password"
            ),
            SerenaEntry(
                path="src/auth/password_utils.py",
                span=(8, 13),
                type=SerenaType.DEF,
                symbol="hash_password",
                sig="def hash_password(password: str) -> str",
                doc="Hash password using bcrypt"
            ),
            SerenaEntry(
                path="src/auth/password_utils.py",
                span=(15, 20),
                type=SerenaType.DEF,
                symbol="verify_password",
                sig="def verify_password(password: str, hashed: str) -> bool", 
                doc="Verify password against hash"
            ),
            SerenaEntry(
                path="src/api/user_controller.py",
                span=(25, 30),
                type=SerenaType.CLASS,
                symbol="UserController",
                sig="class UserController",
                doc="Handle user-related API endpoints"
            ),
            SerenaEntry(
                path="src/api/user_controller.py",
                span=(35, 45),
                type=SerenaType.DEF,
                symbol="login",
                sig="async def login(self, request: LoginRequest) -> LoginResponse",
                doc="Login user and return JWT token"
            ),
            SerenaEntry(
                path="src/database/user_repository.py",
                span=(10, 15),
                type=SerenaType.CLASS,
                symbol="UserRepository",
                sig="class UserRepository",
                doc="Database operations for user management"
            ),
            SerenaEntry(
                path="src/database/user_repository.py",
                span=(18, 22),
                type=SerenaType.DEF,
                symbol="find_by_username",
                sig="async def find_by_username(self, username: str) -> Optional[User]",
                doc="Find user by username"
            ),
            SerenaEntry(
                path="src/utils/jwt_helper.py",
                span=(5, 12),
                type=SerenaType.DEF,
                symbol="generate_jwt_token",
                sig="def generate_jwt_token(user_id: str, secret: str, expires_hours: int = 24) -> str",
                doc="Generate JWT token for user"
            )
        ]
        
        # Create repository map showing relationships
        repo_edges = [
            RepoEdge(source="src/auth/authentication.py", target="src/auth/password_utils.py", weight=0.9),
            RepoEdge(source="src/auth/authentication.py", target="src/database/user_repository.py", weight=0.8),
            RepoEdge(source="src/api/user_controller.py", target="src/auth/authentication.py", weight=0.9),
            RepoEdge(source="src/api/user_controller.py", target="src/utils/jwt_helper.py", weight=0.7),
            RepoEdge(source="tests/test_authentication.py", target="src/auth/authentication.py", weight=0.6),
            RepoEdge(source="tests/test_authentication.py", target="src/auth/password_utils.py", weight=0.5)
        ]
        
        file_ranks = [
            FileRank(path="src/auth/authentication.py", rank=0.95),  # Core auth module
            FileRank(path="src/api/user_controller.py", rank=0.90),  # Main API endpoint
            FileRank(path="src/auth/password_utils.py", rank=0.85),  # Security utilities
            FileRank(path="src/utils/jwt_helper.py", rank=0.80),     # JWT utilities
            FileRank(path="src/database/user_repository.py", rank=0.75), # Data access
            FileRank(path="tests/test_authentication.py", rank=0.60)  # Tests
        ]
        
        return {
            "vector_index": VectorIndex(chunks=vector_chunks),
            "serena_graph": SerenaGraph(entries=serena_entries),
            "repomap": RepoMap(edges=repo_edges, file_ranks=file_ranks),
            "repo_root": "/demo/auth_system",
            "rev": "main",
            "repo_id": "demo_auth_system"
        }
    
    async def initialize(self):
        """Initialize the demo system."""
        print("🚀 Initializing Phase 3 Hybrid Query Engine Demo...")
        
        # Initialize hybrid engine
        await self.hybrid_engine.initialize()
        
        # Initialize performance optimizer
        self.performance_optimizer = create_performance_optimizer(self.hybrid_engine)
        
        print("✅ Phase 3 system initialized successfully!")
        print()
    
    async def demonstrate_query_processing(self):
        """Demonstrate advanced query processing capabilities."""
        print("🧠 Advanced Query Processing Demonstration")
        print("=" * 50)
        
        test_queries = [
            "find function to authenticate users",
            "how does password hashing work",
            "class UserController login method", 
            "JWT token generation implementation",
            "similar patterns to async database queries",
            "bug in authentication flow",
            "where is authenticate_user used"
        ]
        
        for query in test_queries:
            print(f"\n📝 Query: '{query}'")
            processed = self.query_processor.process_query(query)
            
            print(f"   Intent: {processed.intent.value}")
            print(f"   Complexity: {processed.query_complexity:.2f}")
            print(f"   Confidence: {processed.confidence:.2f}")
            
            if processed.entities:
                entities = [f"{e.text}({e.entity_type})" for e in processed.entities[:3]]
                print(f"   Entities: {', '.join(entities)}")
            
            if processed.code_patterns:
                patterns = [p.pattern_type for p in processed.code_patterns[:3]]
                print(f"   Patterns: {', '.join(patterns)}")
            
            if processed.language_hints:
                languages = [lang.value for lang in processed.language_hints[:2]]
                print(f"   Languages: {', '.join(languages)}")
    
    async def demonstrate_search_strategies(self):
        """Demonstrate different search strategies."""
        print("\n🔍 Search Strategy Demonstration")
        print("=" * 50)
        
        test_query = "authentication function implementation"
        strategies = [
            QueryStrategy.VECTOR_FIRST,
            QueryStrategy.SEMANTIC_FIRST,
            QueryStrategy.PARALLEL_HYBRID,
            QueryStrategy.ADAPTIVE
        ]
        
        for strategy in strategies:
            print(f"\n🎯 Testing {strategy.value} strategy...")
            
            context = QueryContext(
                strategy=strategy,
                max_results=5,
                enable_reranking=True
            )
            
            start_time = time.time()
            
            try:
                response = await self.hybrid_engine.search(
                    query=test_query,
                    index_id="demo_index",
                    context=context,
                    **self.demo_search_data
                )
                
                execution_time = (time.time() - start_time) * 1000
                
                print(f"   ⏱️  Execution time: {execution_time:.1f}ms")
                print(f"   📊 Results found: {len(response.results)}")
                
                # Show top results
                for i, result in enumerate(response.results[:3]):
                    print(f"   {i+1}. {result.path} (score: {result.score:.3f})")
                    print(f"      {result.content.text[:100]}...")
            
            except Exception as e:
                print(f"   ❌ Strategy failed: {e}")
    
    async def demonstrate_intelligent_ranking(self):
        """Demonstrate intelligent result ranking."""
        print("\n📈 Intelligent Ranking Demonstration")
        print("=" * 50)
        
        # Test different query types for ranking
        ranking_test_cases = [
            {
                "query": "authentication security implementation",
                "description": "Security-focused query (should boost auth files)"
            },
            {
                "query": "UserController class methods",
                "description": "Class-specific query (should boost exact matches)"
            },
            {
                "query": "password hashing and verification",
                "description": "Functional query (should boost utility functions)"
            }
        ]
        
        for test_case in ranking_test_cases:
            print(f"\n🔎 {test_case['description']}")
            print(f"Query: '{test_case['query']}'")
            
            # Use parallel hybrid for comprehensive results
            context = QueryContext(
                strategy=QueryStrategy.PARALLEL_HYBRID,
                max_results=8,
                enable_reranking=True
            )
            
            response = await self.hybrid_engine.search(
                query=test_case["query"],
                index_id="demo_index",
                context=context,
                **self.demo_search_data
            )
            
            print("   📊 Ranked Results:")
            for i, result in enumerate(response.results[:5]):
                # Show detailed scoring
                vector_score = result.scores.vector
                symbol_score = result.scores.symbol
                graph_score = result.scores.graph
                
                print(f"   {i+1}. {result.path} (final: {result.score:.3f})")
                print(f"      Vector: {vector_score:.3f}, Symbol: {symbol_score:.3f}, Graph: {graph_score:.3f}")
    
    async def demonstrate_performance_optimization(self):
        """Demonstrate performance benchmarking and optimization."""
        print("\n⚡ Performance Optimization Demonstration")
        print("=" * 50)
        
        # Create a focused benchmark suite for demo
        benchmark_suite = create_standard_benchmark_suite()
        benchmark_suite.duration_seconds = 10  # Shorter for demo
        benchmark_suite.strategies_to_test = [
            QueryStrategy.VECTOR_FIRST,
            QueryStrategy.PARALLEL_HYBRID
        ]
        
        print("🏃 Running performance benchmarks...")
        
        try:
            profiles = await self.performance_optimizer.run_comprehensive_benchmark(
                benchmark_suite, self.demo_search_data
            )
            
            print("\n📊 Performance Results:")
            print("-" * 30)
            
            for strategy, profile in profiles.items():
                print(f"\n{strategy.value}:")
                print(f"  Average latency: {profile.avg_response_time_ms:.1f}ms")
                print(f"  P95 latency: {profile.p95_response_time_ms:.1f}ms")
                print(f"  Throughput: {profile.throughput_qps:.1f} QPS")
                print(f"  Cache hit rate: {profile.cache_hit_rate:.1%}")
                print(f"  Accuracy: {profile.accuracy_score:.1%}")
                
                if profile.optimization_recommendations:
                    print("  Recommendations:")
                    for rec in profile.optimization_recommendations[:2]:
                        print(f"    • {rec}")
        
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    async def demonstrate_query_analytics(self):
        """Demonstrate query analytics and insights."""
        print("\n📊 Query Analytics Demonstration")
        print("=" * 50)
        
        # Generate some query activity
        sample_queries = [
            "find authentication function",
            "password verification method",
            "JWT token generation", 
            "user login API endpoint",
            "database user operations"
        ]
        
        print("🔄 Generating query activity...")
        
        for query in sample_queries:
            context = QueryContext(
                strategy=QueryStrategy.ADAPTIVE,
                max_results=3
            )
            
            await self.hybrid_engine.search(
                query=query,
                index_id="demo_index",
                context=context,
                **self.demo_search_data
            )
        
        # Get analytics
        analytics = self.hybrid_engine.get_analytics()
        
        print("\n📈 Query Analytics:")
        print(f"Total queries: {analytics.get('total_queries', 0)}")
        print(f"Cache hit rate: {analytics.get('cache_hit_rate', 0):.1%}")
        
        strategy_usage = analytics.get('strategy_usage', {})
        if strategy_usage:
            print("Strategy usage:")
            for strategy, count in strategy_usage.items():
                print(f"  {strategy}: {count}")
        
        # Response times
        avg_times = analytics.get('avg_response_times', {})
        if avg_times:
            print("Average response times:")
            for strategy, times in avg_times.items():
                if times:
                    avg_time = sum(times) / len(times) if isinstance(times, list) else times
                    print(f"  {strategy}: {avg_time:.1f}ms")
    
    async def run_complete_demo(self):
        """Run the complete Phase 3 demonstration."""
        print("🎯 Phase 3 Hybrid Query Engine - Complete Demonstration")
        print("=" * 60)
        print()
        
        await self.initialize()
        
        try:
            await self.demonstrate_query_processing()
            await self.demonstrate_search_strategies()
            await self.demonstrate_intelligent_ranking()
            await self.demonstrate_query_analytics()
            
            # Only run performance demo if resources allow
            try:
                await self.demonstrate_performance_optimization()
            except Exception as e:
                print(f"⚠️  Performance benchmarking skipped: {e}")
            
            print("\n🎉 Phase 3 Demonstration Complete!")
            print()
            print("Key Features Demonstrated:")
            print("✅ Advanced natural language query processing")
            print("✅ Multiple intelligent search strategies") 
            print("✅ Adaptive strategy selection")
            print("✅ Intelligent result ranking and synthesis")
            print("✅ Real-time query analytics")
            print("✅ Performance benchmarking and optimization")
            print()
            print("Phase 3 successfully combines Lens speed with Mimir intelligence!")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            raise


async def main():
    """Main demo entry point."""
    demo = Phase3Demo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())