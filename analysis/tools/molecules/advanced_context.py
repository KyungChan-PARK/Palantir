"""
Advanced Context Optimization Module

This module provides advanced context optimization techniques for the
document management system, including semantic chunking, dynamic weighting,
and personalized context generation.
"""

import os
import re
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.util import ngrams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'advanced_context.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('advanced_context')

class AdvancedContextOptimizer:
    """Class for advanced context optimization."""
    
    def __init__(self, base_dir='C:\\Users\\packr\\OneDrive\\palantir'):
        """Initialize the optimizer.
        
        Args:
            base_dir (str): Base project directory
        """
        self.base_dir = base_dir
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.persona_profiles = self._load_persona_profiles()
        
    def _load_persona_profiles(self):
        """Load persona profiles from configuration.
        
        Returns:
            dict: Persona profiles
        """
        profiles_path = os.path.join(self.base_dir, 'config', 'persona_profiles.json')
        
        if os.path.exists(profiles_path):
            try:
                with open(profiles_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading persona profiles: {e}")
        
        # Default persona profiles
        return {
            "technical": {
                "keywords": ["implementation", "architecture", "code", "algorithm", "system", 
                             "technical", "specification", "data", "schema", "api"],
                "weight": 1.5
            },
            "management": {
                "keywords": ["strategy", "business", "plan", "budget", "timeline", 
                             "objective", "stakeholder", "resource", "risk", "roi"],
                "weight": 1.5
            },
            "marketing": {
                "keywords": ["campaign", "customer", "audience", "market", "brand", 
                             "message", "segment", "channel", "conversion", "engagement"],
                "weight": 1.5
            }
        }
    
    def preprocess_text(self, text):
        """Preprocess text for analysis.
        
        Args:
            text (str): Text to preprocess
            
        Returns:
            list: Processed tokens
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and punctuation
        tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words]
        
        # Stem tokens
        tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def semantic_chunking(self, document, min_chunk_size=50, max_chunk_size=200, overlap=10):
        """Split document into semantic chunks.
        
        Args:
            document (str): Document text
            min_chunk_size (int): Minimum chunk size in words
            max_chunk_size (int): Maximum chunk size in words
            overlap (int): Number of words to overlap between chunks
            
        Returns:
            list: List of semantic chunks
        """
        logger.info(f"Performing semantic chunking on document ({len(document)} chars)")
        
        # First split into sentences
        sentences = sent_tokenize(document)
        
        # Group sentences into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            words = word_tokenize(sentence)
            sentence_size = len(words)
            
            # If adding this sentence exceeds max_chunk_size and we already have content,
            # finalize the current chunk and start a new one
            if current_size + sentence_size > max_chunk_size and current_size >= min_chunk_size:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_tokens = current_chunk[-overlap:] if overlap > 0 else []
                current_chunk = overlap_tokens + [sentence]
                current_size = len(overlap_tokens) + sentence_size
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add the last chunk if it has content
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def calculate_keyword_density(self, text, keywords):
        """Calculate the density of keywords in text.
        
        Args:
            text (str): Text to analyze
            keywords (list): List of keywords
            
        Returns:
            float: Keyword density score
        """
        # Preprocess text
        tokens = self.preprocess_text(text)
        
        # Preprocess keywords
        processed_keywords = [self.stemmer.stem(keyword.lower()) for keyword in keywords]
        
        # Count keyword occurrences
        keyword_count = sum(1 for token in tokens if token in processed_keywords)
        
        # Calculate density (as percentage of tokens)
        density = (keyword_count / len(tokens)) * 100 if tokens else 0
        
        return density
    
    def calculate_topic_relevance(self, chunk, topic_keywords):
        """Calculate relevance of a chunk to a specific topic.
        
        Args:
            chunk (str): Text chunk
            topic_keywords (list): List of topic keywords
            
        Returns:
            float: Topic relevance score
        """
        # Calculate keyword density
        density = self.calculate_keyword_density(chunk, topic_keywords)
        
        # Calculate TF-IDF-like score
        tokens = self.preprocess_text(chunk)
        token_counter = Counter(tokens)
        
        # Process topic keywords
        processed_keywords = [self.stemmer.stem(keyword.lower()) for keyword in topic_keywords]
        
        # Calculate score based on term frequency of keywords
        score = 0
        for keyword in processed_keywords:
            if keyword in token_counter:
                # Term frequency
                tf = token_counter[keyword] / len(tokens)
                # Simple inverse document frequency approximation
                idf = np.log(1 + (1 / (0.1 + token_counter[keyword] / len(tokens))))
                score += tf * idf
        
        # Combine with density for final score
        relevance = (density * 0.4) + (score * 0.6)
        
        return relevance
    
    def detect_semantic_shifts(self, chunks):
        """Detect semantic shifts between chunks to identify natural boundaries.
        
        Args:
            chunks (list): List of text chunks
            
        Returns:
            list: List of boundary scores between chunks (higher means stronger boundary)
        """
        if len(chunks) <= 1:
            return []
        
        # Get keywords for each chunk
        chunk_keywords = []
        for chunk in chunks:
            tokens = self.preprocess_text(chunk)
            # Get top 10 keywords by frequency
            counter = Counter(tokens)
            keywords = [word for word, _ in counter.most_common(10)]
            chunk_keywords.append(keywords)
        
        # Calculate semantic shift between consecutive chunks
        boundary_scores = []
        for i in range(len(chunks) - 1):
            # Calculate similarity based on shared keywords
            set1 = set(chunk_keywords[i])
            set2 = set(chunk_keywords[i + 1])
            
            # Jaccard similarity
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            similarity = intersection / union if union > 0 else 0
            
            # Boundary score is inverse of similarity
            boundary_score = 1 - similarity
            boundary_scores.append(boundary_score)
        
        return boundary_scores
    
    def persona_based_optimization(self, chunks, persona):
        """Optimize chunks based on user persona.
        
        Args:
            chunks (list): List of text chunks
            persona (str): User persona identifier
            
        Returns:
            list: Optimized chunks with weights
        """
        if persona not in self.persona_profiles:
            logger.warning(f"Persona '{persona}' not found. Using default weighting.")
            return [(chunk, 1.0) for chunk in chunks]
        
        profile = self.persona_profiles[persona]
        keywords = profile["keywords"]
        base_weight = profile.get("weight", 1.0)
        
        weighted_chunks = []
        for chunk in chunks:
            relevance = self.calculate_topic_relevance(chunk, keywords)
            # Scale relevance to a weight between 0.5 and 2.0
            weight = 0.5 + (base_weight * relevance / 10)
            weight = min(max(weight, 0.5), 2.0)  # Clamp between 0.5 and 2.0
            weighted_chunks.append((chunk, weight))
        
        return weighted_chunks
    
    def reorder_chunks_for_coherence(self, weighted_chunks):
        """Reorder chunks to improve narrative coherence.
        
        Args:
            weighted_chunks (list): List of tuples (chunk, weight)
            
        Returns:
            list: Reordered chunks with weights
        """
        if len(weighted_chunks) <= 2:
            return weighted_chunks
        
        # Extract chunks and weights
        chunks = [chunk for chunk, _ in weighted_chunks]
        weights = [weight for _, weight in weighted_chunks]
        
        # Get boundary scores
        boundary_scores = self.detect_semantic_shifts(chunks)
        
        # Identify natural sections based on boundary scores
        sections = []
        current_section = [0]  # Start with the first chunk
        
        for i, score in enumerate(boundary_scores):
            if score > 0.7:  # High boundary score indicates a significant semantic shift
                sections.append(current_section)
                current_section = [i + 1]  # Start a new section
            else:
                current_section.append(i + 1)
        
        if current_section:
            sections.append(current_section)
        
        # Sort sections by average weight
        section_weights = []
        for section in sections:
            section_weight = sum(weights[i] for i in section) / len(section)
            section_weights.append((section, section_weight))
        
        # Sort sections by weight (descending)
        sorted_sections = sorted(section_weights, key=lambda x: x[1], reverse=True)
        
        # Flatten sorted sections into chunk indices
        reordered_indices = []
        for section, _ in sorted_sections:
            reordered_indices.extend(section)
        
        # Reorder chunks and weights
        reordered_chunks = [(chunks[i], weights[i]) for i in reordered_indices]
        
        return reordered_chunks
    
    def optimize_context(self, documents, query=None, persona=None, max_tokens=4000):
        """Optimize context based on documents, query, and user persona.
        
        Args:
            documents (list): List of document texts
            query (str, optional): User query
            persona (str, optional): User persona identifier
            max_tokens (int): Maximum tokens in the optimized context
            
        Returns:
            str: Optimized context
        """
        logger.info(f"Optimizing context from {len(documents)} documents")
        
        # Step 1: Semantic chunking
        all_chunks = []
        for doc in documents:
            chunks = self.semantic_chunking(doc)
            all_chunks.extend(chunks)
        
        logger.info(f"Created {len(all_chunks)} total chunks from documents")
        
        # Step 2: Calculate relevance to query (if provided)
        if query:
            query_tokens = self.preprocess_text(query)
            query_keywords = [token for token in query_tokens if len(token) > 2]
            
            # Calculate relevance scores
            relevance_scores = []
            for chunk in all_chunks:
                relevance = self.calculate_topic_relevance(chunk, query_keywords)
                relevance_scores.append(relevance)
            
            # Sort chunks by relevance
            sorted_chunks = [chunk for _, chunk in sorted(
                zip(relevance_scores, all_chunks), key=lambda x: x[0], reverse=True)]
            
            all_chunks = sorted_chunks
            
            logger.info(f"Sorted chunks by relevance to query")
        
        # Step 3: Apply persona-based optimization
        if persona:
            weighted_chunks = self.persona_based_optimization(all_chunks, persona)
            logger.info(f"Applied persona-based weights to chunks")
        else:
            weighted_chunks = [(chunk, 1.0) for chunk in all_chunks]
        
        # Step 4: Reorder for narrative coherence
        weighted_chunks = self.reorder_chunks_for_coherence(weighted_chunks)
        logger.info(f"Reordered chunks for better coherence")
        
        # Step 5: Select chunks to include in context
        selected_chunks = []
        current_tokens = 0
        
        # Sort by weight (descending) for initial selection
        sorted_weighted_chunks = sorted(weighted_chunks, key=lambda x: x[1], reverse=True)
        
        for chunk, weight in sorted_weighted_chunks:
            tokens = word_tokenize(chunk)
            if current_tokens + len(tokens) <= max_tokens:
                selected_chunks.append((chunk, weight))
                current_tokens += len(tokens)
            else:
                # If we can't fit the whole chunk, break
                break
        
        logger.info(f"Selected {len(selected_chunks)} chunks within token limit of {max_tokens}")
        
        # Step 6: Build final context
        # Reorder selected chunks for coherence
        final_chunks = self.reorder_chunks_for_coherence(selected_chunks)
        
        # Join chunks into context
        optimized_context = "\n\n".join(chunk for chunk, _ in final_chunks)
        
        return optimized_context
    
    def save_optimization_results(self, query, original_documents, optimized_context, persona=None):
        """Save optimization results for analysis.
        
        Args:
            query (str): User query
            original_documents (list): Original documents
            optimized_context (str): Optimized context
            persona (str, optional): User persona
            
        Returns:
            str: Path to saved results
        """
        # Create output directory
        output_dir = os.path.join(self.base_dir, 'output', 'decisions', 'context_optimization')
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create result data
        result = {
            'timestamp': timestamp,
            'query': query,
            'persona': persona,
            'original_documents': original_documents,
            'optimized_context': optimized_context,
            'original_token_count': sum(len(word_tokenize(doc)) for doc in original_documents),
            'optimized_token_count': len(word_tokenize(optimized_context))
        }
        
        # Save result
        result_path = os.path.join(output_dir, f"context_optimization_{timestamp}.json")
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Saved optimization results to {result_path}")
        
        return result_path


# Example usage function
def optimize_documents_for_query(documents, query, persona=None, max_tokens=4000):
    """Wrapper function to optimize documents for a query.
    
    Args:
        documents (list): List of document texts
        query (str): User query
        persona (str, optional): User persona identifier
        max_tokens (int): Maximum tokens in the optimized context
        
    Returns:
        str: Optimized context
    """
    optimizer = AdvancedContextOptimizer()
    optimized_context = optimizer.optimize_context(
        documents=documents,
        query=query,
        persona=persona,
        max_tokens=max_tokens
    )
    
    # Save results
    optimizer.save_optimization_results(
        query=query,
        original_documents=documents,
        optimized_context=optimized_context,
        persona=persona
    )
    
    return optimized_context


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced context optimization')
    parser.add_argument('--input', '-i', required=True, help='Input directory with documents')
    parser.add_argument('--query', '-q', required=True, help='Query to optimize for')
    parser.add_argument('--persona', '-p', default=None, help='User persona')
    parser.add_argument('--output', '-o', default=None, help='Output file for optimized context')
    parser.add_argument('--max-tokens', '-m', type=int, default=4000, help='Maximum tokens in context')
    
    args = parser.parse_args()
    
    # Load documents
    documents = []
    for filename in os.listdir(args.input):
        if filename.endswith('.md') or filename.endswith('.txt'):
            with open(os.path.join(args.input, filename), 'r', encoding='utf-8') as f:
                documents.append(f.read())
    
    # Optimize context
    optimizer = AdvancedContextOptimizer()
    optimized_context = optimizer.optimize_context(
        documents=documents,
        query=args.query,
        persona=args.persona,
        max_tokens=args.max_tokens
    )
    
    # Save optimized context
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(optimized_context)
        print(f"Optimized context saved to {args.output}")
    else:
        print(optimized_context)
