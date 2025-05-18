"""
Advanced Context Optimization Test Script

This script tests the advanced context optimization features
by comparing different optimization strategies.
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Add the project root to the Python path
sys.path.append('C:\\Users\\packr\\OneDrive\\palantir')

# Import project modules
from analysis.tools.atoms.generate_test_documents import create_test_document_set
from analysis.tools.molecules.advanced_context import AdvancedContextOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'test_advanced_context.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('test_advanced_context')

class AdvancedContextTester:
    """Test the advanced context optimization features."""
    
    def __init__(self, base_dir='C:\\Users\\packr\\OneDrive\\palantir',
                 test_dir='temp/test_optimization',
                 results_dir='output/reports/optimization'):
        """Initialize the tester.
        
        Args:
            base_dir (str): Base project directory
            test_dir (str): Directory for test documents
            results_dir (str): Directory for test results
        """
        self.base_dir = base_dir
        self.test_dir = os.path.join(base_dir, test_dir)
        self.results_dir = os.path.join(base_dir, results_dir)
        
        # Create directories
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = AdvancedContextOptimizer(base_dir=base_dir)
        
        # Test results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
    
    def setup_test_documents(self, document_count=50):
        """Generate test documents for optimization testing.
        
        Args:
            document_count (int): Number of documents to generate
            
        Returns:
            list: List of document texts
        """
        logger.info(f"Generating {document_count} test documents")
        
        # Create directory for test documents
        test_set_dir = os.path.join(self.test_dir, f"docs_{document_count}")
        if os.path.exists(test_set_dir):
            logger.info(f"Test documents already exist at {test_set_dir}")
        else:
            logger.info(f"Creating test documents at {test_set_dir}")
            metadata_path = os.path.join(self.test_dir, f"metadata_{document_count}.json")
            
            # Generate test documents
            create_test_document_set(
                output_dir=test_set_dir,
                count=document_count,
                metadata_file=metadata_path
            )
        
        # Load documents
        documents = []
        for filename in os.listdir(test_set_dir):
            if filename.endswith('.md'):
                with open(os.path.join(test_set_dir, filename), 'r', encoding='utf-8') as f:
                    documents.append(f.read())
        
        logger.info(f"Loaded {len(documents)} test documents")
        return documents
    
    def test_semantic_chunking(self, documents, chunk_sizes=None):
        """Test semantic chunking with different parameters.
        
        Args:
            documents (list): List of document texts
            chunk_sizes (list): List of chunk sizes to test
            
        Returns:
            dict: Test results for semantic chunking
        """
        if chunk_sizes is None:
            chunk_sizes = [50, 100, 200, 500]
        
        logger.info(f"Testing semantic chunking with sizes: {chunk_sizes}")
        
        results = []
        
        for min_chunk_size in chunk_sizes:
            max_chunk_size = min_chunk_size * 2
            
            logger.info(f"Testing chunk size: min={min_chunk_size}, max={max_chunk_size}")
            
            start_time = time.time()
            
            # Process a sample of documents
            sample_docs = documents[:5]
            chunks = []
            
            for doc in sample_docs:
                doc_chunks = self.optimizer.semantic_chunking(
                    document=doc,
                    min_chunk_size=min_chunk_size,
                    max_chunk_size=max_chunk_size
                )
                chunks.extend(doc_chunks)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate stats
            chunk_lengths = [len(chunk.split()) for chunk in chunks]
            avg_length = sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0
            
            result = {
                'min_chunk_size': min_chunk_size,
                'max_chunk_size': max_chunk_size,
                'processing_time_seconds': processing_time,
                'total_chunks': len(chunks),
                'avg_chunk_length': avg_length,
                'min_length': min(chunk_lengths) if chunk_lengths else 0,
                'max_length': max(chunk_lengths) if chunk_lengths else 0
            }
            
            results.append(result)
            logger.info(f"Chunking result: {result}")
        
        return results
    
    def test_persona_optimization(self, documents, personas=None):
        """Test persona-based optimization.
        
        Args:
            documents (list): List of document texts
            personas (list): List of personas to test
            
        Returns:
            dict: Test results for persona optimization
        """
        if personas is None:
            personas = ['technical', 'management', 'marketing']
        
        logger.info(f"Testing persona optimization for personas: {personas}")
        
        results = []
        
        # Create sample documents for testing
        sample_docs = documents[:3]
        
        for persona in personas:
            logger.info(f"Testing optimization for persona: {persona}")
            
            start_time = time.time()
            
            # Create chunks
            all_chunks = []
            for doc in sample_docs:
                chunks = self.optimizer.semantic_chunking(doc)
                all_chunks.extend(chunks)
            
            # Apply persona optimization
            weighted_chunks = self.optimizer.persona_based_optimization(all_chunks, persona)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate stats
            weights = [weight for _, weight in weighted_chunks]
            
            result = {
                'persona': persona,
                'processing_time_seconds': processing_time,
                'chunk_count': len(weighted_chunks),
                'avg_weight': sum(weights) / len(weights) if weights else 0,
                'min_weight': min(weights) if weights else 0,
                'max_weight': max(weights) if weights else 0
            }
            
            results.append(result)
            logger.info(f"Persona optimization result: {result}")
        
        return results
    
    def test_query_optimization(self, documents, queries):
        """Test query-based optimization.
        
        Args:
            documents (list): List of document texts
            queries (list): List of queries to test
            
        Returns:
            dict: Test results for query optimization
        """
        logger.info(f"Testing query optimization for queries: {queries}")
        
        results = []
        
        # Create sample documents for testing
        sample_docs = documents[:5]
        
        for query in queries:
            logger.info(f"Testing optimization for query: {query}")
            
            start_time = time.time()
            
            # Optimize context for query
            optimized_context = self.optimizer.optimize_context(
                documents=sample_docs,
                query=query
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Calculate stats
            original_tokens = sum(len(doc.split()) for doc in sample_docs)
            optimized_tokens = len(optimized_context.split())
            
            result = {
                'query': query,
                'processing_time_seconds': processing_time,
                'original_tokens': original_tokens,
                'optimized_tokens': optimized_tokens,
                'compression_ratio': optimized_tokens / original_tokens if original_tokens > 0 else 0
            }
            
            results.append(result)
            logger.info(f"Query optimization result: {result}")
        
        return results
    
    def test_end_to_end(self, documents, queries, personas):
        """Test end-to-end context optimization.
        
        Args:
            documents (list): List of document texts
            queries (list): List of queries to test
            personas (list): List of personas to test
            
        Returns:
            dict: Test results for end-to-end optimization
        """
        logger.info("Testing end-to-end context optimization")
        
        results = []
        
        # Test different combinations
        for query in queries:
            for persona in personas:
                logger.info(f"Testing optimization for query '{query}' and persona '{persona}'")
                
                start_time = time.time()
                
                # Optimize context
                optimized_context = self.optimizer.optimize_context(
                    documents=documents[:5],
                    query=query,
                    persona=persona
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Calculate stats
                original_tokens = sum(len(doc.split()) for doc in documents[:5])
                optimized_tokens = len(optimized_context.split())
                
                result = {
                    'query': query,
                    'persona': persona,
                    'processing_time_seconds': processing_time,
                    'original_tokens': original_tokens,
                    'optimized_tokens': optimized_tokens,
                    'compression_ratio': optimized_tokens / original_tokens if original_tokens > 0 else 0
                }
                
                results.append(result)
                logger.info(f"End-to-end optimization result: {result}")
        
        return results
    
    def run_all_tests(self):
        """Run all optimization tests.
        
        Returns:
            dict: Test results
        """
        logger.info("Running all optimization tests")
        
        # Generate test documents
        documents = self.setup_test_documents(document_count=50)
        
        # Define test parameters
        chunk_sizes = [50, 100, 200, 500]
        personas = ['technical', 'management', 'marketing']
        queries = [
            "How to improve system performance?",
            "What are the business impacts of the new strategy?",
            "How effective is our marketing campaign?"
        ]
        
        # Run tests
        chunking_results = self.test_semantic_chunking(documents, chunk_sizes)
        persona_results = self.test_persona_optimization(documents, personas)
        query_results = self.test_query_optimization(documents, queries)
        end_to_end_results = self.test_end_to_end(documents, queries, personas)
        
        # Combine results
        self.results['tests'] = {
            'semantic_chunking': chunking_results,
            'persona_optimization': persona_results,
            'query_optimization': query_results,
            'end_to_end': end_to_end_results
        }
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(self.results_dir, f"optimization_test_results_{timestamp}.json")
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Test results saved to {results_file}")
        
        # Generate visualizations
        self.generate_visualizations(self.results, self.results_dir, timestamp)
        
        return self.results
    
    def generate_visualizations(self, results, output_dir, timestamp):
        """Generate visualizations of test results.
        
        Args:
            results (dict): Test results
            output_dir (str): Output directory for visualizations
            timestamp (str): Timestamp for filenames
        """
        logger.info("Generating visualizations of test results")
        
        # Create a directory for visualizations
        viz_dir = os.path.join(output_dir, f"visualizations_{timestamp}")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Semantic Chunking Performance
        try:
            chunking_results = results['tests']['semantic_chunking']
            if chunking_results:
                df = pd.DataFrame(chunking_results)
                
                plt.figure(figsize=(10, 6))
                plt.plot(df['min_chunk_size'], df['processing_time_seconds'], marker='o')
                plt.title('Semantic Chunking Processing Time vs. Chunk Size')
                plt.xlabel('Minimum Chunk Size (words)')
                plt.ylabel('Processing Time (seconds)')
                plt.grid(True)
                plt.savefig(os.path.join(viz_dir, 'chunking_time.png'))
                
                plt.figure(figsize=(10, 6))
                plt.plot(df['min_chunk_size'], df['total_chunks'], marker='o')
                plt.title('Number of Chunks vs. Chunk Size')
                plt.xlabel('Minimum Chunk Size (words)')
                plt.ylabel('Number of Chunks')
                plt.grid(True)
                plt.savefig(os.path.join(viz_dir, 'chunk_count.png'))
                
                plt.figure(figsize=(10, 6))
                plt.plot(df['min_chunk_size'], df['avg_chunk_length'], marker='o')
                plt.title('Average Chunk Length vs. Minimum Chunk Size')
                plt.xlabel('Minimum Chunk Size (words)')
                plt.ylabel('Average Chunk Length (words)')
                plt.grid(True)
                plt.savefig(os.path.join(viz_dir, 'avg_chunk_length.png'))
                
                logger.info("Generated semantic chunking visualizations")
        except Exception as e:
            logger.error(f"Error generating chunking visualizations: {e}")
        
        # Persona Optimization Performance
        try:
            persona_results = results['tests']['persona_optimization']
            if persona_results:
                df = pd.DataFrame(persona_results)
                
                plt.figure(figsize=(10, 6))
                plt.bar(df['persona'], df['avg_weight'])
                plt.title('Average Chunk Weight by Persona')
                plt.xlabel('Persona')
                plt.ylabel('Average Weight')
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(viz_dir, 'persona_weights.png'))
                
                plt.figure(figsize=(10, 6))
                plt.bar(df['persona'], df['processing_time_seconds'])
                plt.title('Persona Optimization Processing Time')
                plt.xlabel('Persona')
                plt.ylabel('Processing Time (seconds)')
                plt.grid(True, axis='y')
                plt.savefig(os.path.join(viz_dir, 'persona_time.png'))
                
                logger.info("Generated persona optimization visualizations")
        except Exception as e:
            logger.error(f"Error generating persona visualizations: {e}")
        
        # Query Optimization Performance
        try:
            query_results = results['tests']['query_optimization']
            if query_results:
                df = pd.DataFrame(query_results)
                
                plt.figure(figsize=(12, 6))
                plt.bar(df['query'], df['compression_ratio'])
                plt.title('Compression Ratio by Query')
                plt.xlabel('Query')
                plt.ylabel('Compression Ratio')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'query_compression.png'))
                
                plt.figure(figsize=(12, 6))
                plt.bar(df['query'], df['processing_time_seconds'])
                plt.title('Query Optimization Processing Time')
                plt.xlabel('Query')
                plt.ylabel('Processing Time (seconds)')
                plt.xticks(rotation=45, ha='right')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'query_time.png'))
                
                logger.info("Generated query optimization visualizations")
        except Exception as e:
            logger.error(f"Error generating query visualizations: {e}")
        
        # End-to-End Performance
        try:
            end_to_end_results = results['tests']['end_to_end']
            if end_to_end_results:
                df = pd.DataFrame(end_to_end_results)
                
                plt.figure(figsize=(12, 8))
                pivot_df = df.pivot(index='query', columns='persona', values='compression_ratio')
                pivot_df.plot(kind='bar', figsize=(12, 8))
                plt.title('Compression Ratio by Query and Persona')
                plt.xlabel('Query')
                plt.ylabel('Compression Ratio')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'end_to_end_compression.png'))
                
                plt.figure(figsize=(12, 8))
                pivot_time_df = df.pivot(index='query', columns='persona', values='processing_time_seconds')
                pivot_time_df.plot(kind='bar', figsize=(12, 8))
                plt.title('Processing Time by Query and Persona')
                plt.xlabel('Query')
                plt.ylabel('Processing Time (seconds)')
                plt.grid(True, axis='y')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'end_to_end_time.png'))
                
                logger.info("Generated end-to-end optimization visualizations")
        except Exception as e:
            logger.error(f"Error generating end-to-end visualizations: {e}")
        
        logger.info(f"Visualizations saved to {viz_dir}")


def main():
    """Main function for running optimization tests."""
    parser = argparse.ArgumentParser(description='Test advanced context optimization')
    parser.add_argument('--test-dir', '-t', default='temp/test_optimization',
                      help='Directory for test documents')
    parser.add_argument('--results-dir', '-r', default='output/reports/optimization',
                      help='Directory for test results')
    
    args = parser.parse_args()
    
    # Run tests
    tester = AdvancedContextTester(
        test_dir=args.test_dir,
        results_dir=args.results_dir
    )
    
    try:
        results = tester.run_all_tests()
        
        # Print summary
        print("\nAdvanced Context Optimization Test Summary:")
        print("==========================================")
        
        # Semantic Chunking
        print("\nSemantic Chunking Performance:")
        for result in results['tests']['semantic_chunking']:
            print(f"  Min Size: {result['min_chunk_size']} words, " +
                 f"Chunks: {result['total_chunks']}, " +
                 f"Avg Length: {result['avg_chunk_length']:.2f} words, " +
                 f"Time: {result['processing_time_seconds']:.4f} seconds")
        
        # Persona Optimization
        print("\nPersona Optimization Performance:")
        for result in results['tests']['persona_optimization']:
            print(f"  Persona: {result['persona']}, " +
                 f"Avg Weight: {result['avg_weight']:.4f}, " +
                 f"Time: {result['processing_time_seconds']:.4f} seconds")
        
        # Query Optimization
        print("\nQuery Optimization Performance:")
        for result in results['tests']['query_optimization']:
            print(f"  Query: '{result['query']}', " +
                 f"Compression: {result['compression_ratio']:.4f}, " +
                 f"Time: {result['processing_time_seconds']:.4f} seconds")
        
        print("\nResults saved to directory:", args.results_dir)
    
    except Exception as e:
        logger.error(f"Error running optimization tests: {e}")
        raise


if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs(os.path.join('C:\\Users\\packr\\OneDrive\\palantir', 'logs'), exist_ok=True)
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Error in advanced context test: {e}")
        raise
