"""
Hierarchical & Chunked Embeddings for Large Codebases

This module provides memory-efficient embedding of large codebases by:
1. Breaking files into functions/classes
2. Using windowed encoding for large blocks
3. Implementing fine-grained caching
4. Streaming pipeline for batch processing
"""

import os
import hashlib
import sqlite3
import h5py
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Iterator
from dataclasses import dataclass
from pathlib import Path
import ast
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .encoder import encode_ast, HybridEmbeddingConfig
from dsl.parser import parse_program

@dataclass
class FunctionInfo:
    """Information about a function/class in a file."""
    name: str
    start_line: int
    end_line: int
    ast_nodes: List[Any]
    fingerprint: str
    embedding: Optional[np.ndarray] = None

@dataclass
class FileInfo:
    """Information about a file in the codebase."""
    path: str
    fingerprint: str
    functions: List[FunctionInfo]
    file_embedding: Optional[np.ndarray] = None
    last_modified: float = 0.0

@dataclass
class ChunkedEmbeddingConfig:
    """Configuration for chunked embedding system."""
    vector_dim: int = 128
    max_function_lines: int = 500
    max_function_tokens: int = 1000
    window_size: int = 512
    window_stride: int = 256
    cache_dir: str = "embeddings_cache"
    max_ram_gb: float = 2.0
    batch_size: int = 10
    use_parallel: bool = True
    max_workers: int = 4

class HierarchicalEmbedder:
    """
    Memory-efficient hierarchical embedder for large codebases.
    
    Features:
    - Function-level chunking
    - Windowed encoding for large blocks
    - Fine-grained caching
    - Streaming pipeline
    - Memory usage monitoring
    """
    
    def __init__(self, config: Optional[ChunkedEmbeddingConfig] = None):
        self.config = config or ChunkedEmbeddingConfig()
        self.encoder_config = HybridEmbeddingConfig(
            vector_dim=self.config.vector_dim,
            use_neural_embeddings=False,  # Keep it simple for memory efficiency
            hash_weight=0.4,
            structural_weight=0.4,
            semantic_weight=0.2
        )
        
        # Setup cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache database
        self._init_cache_db()
        
        # Memory tracking
        self.peak_memory_mb = 0
        self.current_memory_mb = 0
        
        logging.info(f"Initialized HierarchicalEmbedder with {self.config.vector_dim}D vectors")
        logging.info(f"Cache directory: {self.cache_dir}")
        logging.info(f"Max RAM: {self.config.max_ram_gb}GB")
    
    def _init_cache_db(self):
        """Initialize SQLite cache database."""
        db_path = self.cache_dir / "embeddings_cache.db"
        # Don't create connection here - create per thread
        self.db_path = str(db_path)
        
        # Create tables if they don't exist
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                file_embedding BLOB,
                last_modified REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                file_path TEXT,
                function_name TEXT,
                start_line INTEGER,
                end_line INTEGER,
                fingerprint TEXT,
                embedding BLOB,
                PRIMARY KEY (file_path, function_name),
                FOREIGN KEY (file_path) REFERENCES files (path)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_connection(self):
        """Get a thread-safe database connection."""
        return sqlite3.connect(self.db_path)
    
    def _compute_fingerprint(self, content: str) -> str:
        """Compute SHA-256 fingerprint of content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _extract_functions(self, file_path: str) -> List[FunctionInfo]:
        """Extract functions and classes from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse with ast module for better function detection
            tree = ast.parse(content)
            
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Extract function/class info
                    func_name = node.name
                    start_line = node.lineno
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') and node.end_lineno is not None else start_line
                    
                    # Extract the function/class code
                    lines = content.split('\n')
                    func_lines = lines[start_line-1:end_line]
                    func_content = '\n'.join(func_lines)
                    
                    # Parse with our DSL parser for AST nodes
                    try:
                        ast_nodes = parse_program(func_content)
                    except:
                        # Fallback: create simple AST representation
                        ast_nodes = [("function_def", func_name, [])]
                    
                    # Compute fingerprint
                    fingerprint = self._compute_fingerprint(func_content)
                    
                    functions.append(FunctionInfo(
                        name=func_name,
                        start_line=start_line,
                        end_line=end_line,
                        ast_nodes=ast_nodes,
                        fingerprint=fingerprint
                    ))
            
            return functions
            
        except Exception as e:
            logging.warning(f"Failed to extract functions from {file_path}: {e}")
            return []
    
    def _should_chunk_function(self, func: FunctionInfo) -> bool:
        """Determine if a function should be chunked."""
        line_count = func.end_line - func.start_line + 1
        return line_count > self.config.max_function_lines
    
    def _chunk_function(self, func: FunctionInfo) -> List[FunctionInfo]:
        """Chunk a large function into smaller windows."""
        if not self._should_chunk_function(func):
            return [func]
        
        # For now, create simple chunks based on AST nodes
        # In a full implementation, this would use token-based windowing
        ast_nodes = func.ast_nodes
        chunks = []
        
        for i in range(0, len(ast_nodes), self.config.window_stride):
            chunk_nodes = ast_nodes[i:i + self.config.window_size]
            if not chunk_nodes:
                continue
            
            chunk_content = str(chunk_nodes)  # Simplified
            chunk_fingerprint = self._compute_fingerprint(chunk_content)
            
            chunks.append(FunctionInfo(
                name=f"{func.name}_chunk_{i//self.config.window_stride}",
                start_line=func.start_line + i,
                end_line=func.start_line + min(i + self.config.window_size, len(ast_nodes)),
                ast_nodes=chunk_nodes,
                fingerprint=chunk_fingerprint
            ))
        
        return chunks
    
    def _embed_function(self, func: FunctionInfo) -> np.ndarray:
        """Embed a single function."""
        try:
            # Check cache first
            cached_embedding = self._get_cached_embedding(func.fingerprint)
            if cached_embedding is not None:
                return cached_embedding
            
            # Encode the function
            embedding = encode_ast(func.ast_nodes, self.config.vector_dim, self.encoder_config)
            
            # Cache the result
            self._cache_embedding(func.fingerprint, embedding)
            
            return embedding
            
        except Exception as e:
            logging.error(f"Failed to embed function {func.name}: {e}")
            # Return zero vector as fallback
            return np.zeros(self.config.vector_dim)
    
    def _get_cached_embedding(self, fingerprint: str) -> Optional[np.ndarray]:
        """Get cached embedding from database."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT embedding FROM functions WHERE fingerprint = ?",
            (fingerprint,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0]:
            return np.frombuffer(result[0], dtype=np.float64)
        return None
    
    def _cache_embedding(self, fingerprint: str, embedding: np.ndarray):
        """Cache embedding in database."""
        conn = self._get_connection()
        embedding_blob = embedding.astype(np.float64).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO functions (fingerprint, embedding) VALUES (?, ?)",
            (fingerprint, embedding_blob)
        )
        conn.commit()
        conn.close()
    
    def _update_memory_usage(self, additional_mb: float = 0):
        """Track memory usage."""
        self.current_memory_mb += additional_mb
        self.peak_memory_mb = max(self.peak_memory_mb, self.current_memory_mb)
        
        if self.current_memory_mb > self.config.max_ram_gb * 1024:
            logging.warning(f"Memory usage high: {self.current_memory_mb:.1f}MB")
    
    def embed_file(self, file_path: str) -> FileInfo:
        """Embed a single file with function-level chunking."""
        logging.info(f"Embedding file: {file_path}")
        
        # Check if file needs re-embedding
        file_fingerprint = self._compute_fingerprint(Path(file_path).read_text())
        last_modified = os.path.getmtime(file_path)
        
        # Check cache
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT file_embedding, last_modified FROM files WHERE path = ? AND fingerprint = ?",
            (file_path, file_fingerprint)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] and result[1] >= last_modified:
            # Use cached result
            file_embedding = np.frombuffer(result[0], dtype=np.float64)
            logging.info(f"Using cached embedding for {file_path}")
            return FileInfo(
                path=file_path,
                fingerprint=file_fingerprint,
                functions=[],  # We don't need to reconstruct functions for cached files
                file_embedding=file_embedding,
                last_modified=last_modified
            )
        
        # Extract functions
        functions = self._extract_functions(file_path)
        logging.info(f"Found {len(functions)} functions in {file_path}")
        
        # Chunk large functions
        all_chunks = []
        for func in functions:
            chunks = self._chunk_function(func)
            all_chunks.extend(chunks)
        
        logging.info(f"After chunking: {len(all_chunks)} chunks")
        
        # Embed functions/chunks
        function_embeddings = []
        for chunk in all_chunks:
            embedding = self._embed_function(chunk)
            function_embeddings.append(embedding)
            chunk.embedding = embedding
            
            # Update memory usage (rough estimate: 128-dim float64 = 1KB)
            self._update_memory_usage(0.001)
        
        # Pool function embeddings into file embedding
        if function_embeddings:
            # Ensure all embeddings have the same shape
            valid_embeddings = []
            for emb in function_embeddings:
                if emb is not None and emb.shape == (self.config.vector_dim,):
                    valid_embeddings.append(emb)
            
            if valid_embeddings:
                file_embedding = np.mean(valid_embeddings, axis=0)
                # Normalize
                norm = np.linalg.norm(file_embedding)
                if norm > 0:
                    file_embedding = file_embedding / norm
            else:
                file_embedding = np.zeros(self.config.vector_dim)
        else:
            file_embedding = np.zeros(self.config.vector_dim)
        
        # Cache file embedding
        conn = self._get_connection()
        embedding_blob = file_embedding.astype(np.float64).tobytes()
        conn.execute(
            "INSERT OR REPLACE INTO files (path, fingerprint, file_embedding, last_modified) VALUES (?, ?, ?, ?)",
            (file_path, file_fingerprint, embedding_blob, last_modified)
        )
        conn.commit()
        conn.close()
        
        return FileInfo(
            path=file_path,
            fingerprint=file_fingerprint,
            functions=all_chunks,
            file_embedding=file_embedding,
            last_modified=last_modified
        )
    
    def embed_directory(self, directory: str, file_pattern: str = "*.py") -> Iterator[FileInfo]:
        """Embed all files in a directory using streaming pipeline."""
        directory_path = Path(directory)
        python_files = list(directory_path.rglob(file_pattern))
        
        logging.info(f"Found {len(python_files)} Python files in {directory}")
        
        if self.config.use_parallel:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {
                    executor.submit(self.embed_file, str(file_path)): file_path 
                    for file_path in python_files
                }
                
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_info = future.result()
                        yield file_info
                    except Exception as e:
                        logging.error(f"Failed to embed {file_path}: {e}")
        else:
            # Sequential processing
            for file_path in python_files:
                try:
                    file_info = self.embed_file(str(file_path))
                    yield file_info
                except Exception as e:
                    logging.error(f"Failed to embed {file_path}: {e}")
    
    def save_embeddings(self, file_infos: List[FileInfo], output_path: str):
        """Save embeddings to HDF5 file."""
        with h5py.File(output_path, 'w') as f:
            # Save file embeddings
            file_embeddings = np.array([fi.file_embedding for fi in file_infos if fi.file_embedding is not None])
            f.create_dataset('file_embeddings', data=file_embeddings)
            
            # Save file paths - ensure they are strings
            file_paths = [str(fi.path) for fi in file_infos]
            f.create_dataset('file_paths', data=np.array(file_paths, dtype=h5py.special_dtype(vlen=str)))
            
            # Save function embeddings
            all_function_embeddings = []
            all_function_names = []
            all_file_paths = []
            
            for fi in file_infos:
                for func in fi.functions:
                    if func.embedding is not None and func.embedding.shape == (self.config.vector_dim,):
                        all_function_embeddings.append(func.embedding)
                        all_function_names.append(func.name)
                        all_file_paths.append(str(fi.path))
            
            if all_function_embeddings:
                # Ensure all embeddings have the same shape
                embeddings_array = np.array(all_function_embeddings)
                f.create_dataset('function_embeddings', data=embeddings_array)
                f.create_dataset('function_names', data=np.array(all_function_names, dtype=h5py.special_dtype(vlen=str)))
                f.create_dataset('function_file_paths', data=np.array(all_file_paths, dtype=h5py.special_dtype(vlen=str)))
        
        logging.info(f"Saved embeddings to {output_path}")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return {
            'current_memory_mb': self.current_memory_mb,
            'peak_memory_mb': self.peak_memory_mb,
            'max_allowed_mb': self.config.max_ram_gb * 1024
        }
    
    def clear_cache(self):
        """Clear the embedding cache."""
        conn = self._get_connection()
        conn.execute("DELETE FROM files")
        conn.execute("DELETE FROM functions")
        conn.commit()
        conn.close()
        logging.info("Cleared embedding cache")
    
    def close(self):
        """Close database connection."""
        # No explicit close needed here as connections are managed per thread
        pass 