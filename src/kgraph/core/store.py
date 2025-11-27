import sqlite3
import lancedb
import json
import os
import threading
import logging
import sys
import contextlib
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphStore:
    def __init__(self, db_path: str = "kgraph.db", lancedb_path: str = "lancedb_data"):
        self.db_path = db_path
        self.lancedb_path = lancedb_path
        self._local = threading.local()
        self.table = None
        self.encoder = None
        self.mlx_model = None
        self.mlx_tokenizer = None
        
        self._init_sqlite()
        self._init_encoder()
        
        self.lance_db = lancedb.connect(self.lancedb_path)
        self._init_lancedb()

    def _init_sqlite(self):
        """Initialize the SQLite database schema."""
        # Initialize DB schema (using a temporary connection)
        schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;") # Enable Write-Ahead Logging for concurrency
            with open(schema_path, 'r') as f:
                conn.executescript(f.read())

    def _init_encoder(self):
        """Initialize the vector embedding encoder (MLX or SentenceTransformers)."""
        # Suppress stdout and logging during encoder init
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        
        # Try to use MLX for Apple Silicon (much faster)
        self.use_mlx = False
        try:
            from mlx_embeddings import load, generate
            logger.info("Using MLX backend (Apple Neural Engine)")
            with contextlib.redirect_stdout(sys.stderr):
                self.mlx_model, self.mlx_tokenizer = load("mlx-community/all-MiniLM-L6-v2-4bit")
                self.mlx_generate = generate
            self.use_mlx = True
        except ImportError:
            logger.info("MLX not available, falling back to sentence-transformers")
            self._init_fallback_encoder()
        except Exception as e:
            logger.warning(f"Error loading MLX ({e}), falling back to sentence-transformers")
            self._init_fallback_encoder()

    def _init_fallback_encoder(self):
        with contextlib.redirect_stdout(sys.stderr):
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def _get_conn(self):
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_lancedb(self):
        # Define schema implicitly by creating table with first item or checking existence
        try:
            self.table = self.lance_db.open_table("code_vectors")
            logger.info(f"Opened existing LanceDB table at {self.lancedb_path}")
        except Exception as e:
            # Table will be created on first insertion
            logger.info(f"LanceDB table not found, will be created on first indexing")
            self.table = None

    def add_node(self, node_id: str, node_type: str, name: str, file_path: str, 
                 start_line: int, end_line: int, properties: Dict[str, Any] = {}, 
                 code_content: str = ""):
        
        # 1. Store in SQLite
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO nodes (id, type, name, file_path, start_line, end_line, properties)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (node_id, node_type, name, file_path, start_line, end_line, json.dumps(properties)))
        conn.commit()

        # 2. Vector embeddings are now handled in batch mode (see add_nodes_batch)
        # This is much faster than encoding one at a time
        
    def add_nodes_batch(self, nodes: List[Dict[str, Any]]):
        """Add multiple nodes at once with batch embedding computation."""
        if not nodes:
            return
            
        conn = self._get_conn()
        cursor = conn.cursor()
        
        # Collect nodes with code content for batch embedding
        nodes_with_code = []
        code_contents = []
        
        for node in nodes:
            # Insert into SQLite
            cursor.execute("""
                INSERT OR REPLACE INTO nodes (id, type, name, file_path, start_line, end_line, properties)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                node['node_id'], 
                node['node_type'], 
                node['name'], 
                node['file_path'], 
                node['start_line'], 
                node['end_line'], 
                json.dumps(node.get('properties', {}))
            ))
            
            # Collect for batch embedding
            if node.get('code_content'):
                nodes_with_code.append(node)
                code_contents.append(node['code_content'])
        
        conn.commit()
        
        # Batch encode all code snippets at once (MUCH faster)
        if code_contents:
            import sys
            print(f"[Embedding] Processing {len(code_contents)} code snippets...", file=sys.stderr, flush=True)
            
            if self.use_mlx:
                # MLX encoder (Apple Neural Engine - very fast!)
                output = self.mlx_generate(self.mlx_model, self.mlx_tokenizer, code_contents)
                # Mean pooling or CLS token? SentenceTransformers usually does mean pooling.
                # But for simplicity and speed, let's use CLS token (index 0) if available, 
                # or mean pooling if we want to be more accurate.
                # Let's use mean pooling to match SentenceTransformer behavior better.
                # Actually, all-MiniLM-L6-v2 uses mean pooling.
                # output.last_hidden_state is (batch, seq_len, hidden_dim)
                # We need to average over seq_len, ignoring padding.
                # For now, let's just take the CLS token ([:, 0, :]) which is a decent approximation
                # and very fast.
                embeddings = output.last_hidden_state[:, 0, :]
            else:
                # Sentence-transformers (CPU/MPS fallback)
                embeddings = self.encoder.encode(code_contents, show_progress_bar=False, batch_size=128)
                
            print(f"[Embedding] Completed embedding computation", file=sys.stderr, flush=True)
            
            # Prepare data for LanceDB
            lance_data = []
            for i, node in enumerate(nodes_with_code):
                # Convert embedding to list (handle both MLX arrays and numpy arrays)
                embedding_list = embeddings[i].tolist() if hasattr(embeddings[i], 'tolist') else list(embeddings[i])
                
                lance_data.append({
                    "id": node['node_id'],
                    "vector": embedding_list,
                    "type": node['node_type'],
                    "name": node['name'],
                    "file_path": node['file_path'],
                    "text": node['code_content']
                })
            
            # Insert into LanceDB
            # Insert into LanceDB with robust error handling
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.table is None:
                        try:
                            # Try to open first (in case another process created it)
                            self.table = self.lance_db.open_table("code_vectors")
                            self.table.add(lance_data)
                        except Exception:
                            # Table doesn't exist, try to create
                            try:
                                self.table = self.lance_db.create_table("code_vectors", lance_data)
                            except Exception as e:
                                if "already exists" in str(e):
                                    # Race condition: another process created it
                                    self.table = self.lance_db.open_table("code_vectors")
                                    self.table.add(lance_data)
                                else:
                                    raise e
                    else:
                        self.table.add(lance_data)
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to write to LanceDB after {max_retries} attempts: {e}")
                        # Don't raise, just log error to avoid crashing the whole indexing process
                        # But this means vector search will be incomplete
                    else:
                        import time
                        time.sleep(0.1 * (attempt + 1)) # Exponential backoff

    def add_edge(self, source_id: str, target_id: str, edge_type: str, properties: Dict[str, Any] = {}):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO edges (source_id, target_id, type, properties)
            VALUES (?, ?, ?, ?)
        """, (source_id, target_id, edge_type, json.dumps(properties)))
        conn.commit()

    def search_nodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if self.table is None:
            # Try to open table again in case it was created after initialization
            try:
                self.table = self.lance_db.open_table("code_vectors")
                logger.info("Successfully opened LanceDB table")
            except Exception as e:
                logger.error(f"LanceDB table 'code_vectors' not found at {self.lancedb_path}. Please reindex the codebase first.")
                raise ValueError(f"Vector database not initialized. Please run reindex_codebase first. Error: {e}")
        
        try:
            if self.use_mlx:
                output = self.mlx_generate(self.mlx_model, self.mlx_tokenizer, [query])
                query_vector = output.last_hidden_state[:, 0, :][0].tolist()
            else:
                query_vector = self.encoder.encode(query).tolist()
                
            results = self.table.search(query_vector).limit(limit).to_list()
            return results
        except Exception as e:
            logger.error(f"Error searching LanceDB: {e}")
            if "No such file or directory" in str(e) or "corrupt" in str(e).lower():
                raise ValueError(f"Vector database corrupted or missing. Please run 'reindex_codebase' to fix it. Error: {e}")
            raise ValueError(f"Search failed: {e}")

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
        row = cursor.fetchone()
        if row:
            d = dict(row)
            d['properties'] = json.loads(d['properties']) if d['properties'] else {}
            return d
        return None

    def get_related(self, node_id: str, edge_type: str = None, direction: str = "out") -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if direction == "out":
            query = "SELECT target_id as id, type, properties FROM edges WHERE source_id = ?"
            params = [node_id]
        elif direction == "in":
            query = "SELECT source_id as id, type, properties FROM edges WHERE target_id = ?"
            params = [node_id]
        else:
            raise ValueError("Direction must be 'in' or 'out'")

        if edge_type:
            query += " AND type = ?"
            params.append(edge_type)

        cursor.execute(query, params)
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d['properties'] = json.loads(d['properties']) if d['properties'] else {}
            
            # Skip UNKNOWN targets - they haven't been resolved yet
            if d['id'].startswith('UNKNOWN:'):
                continue
                
            # Fetch full node details for the related node
            node_details = self.get_node(d['id'])
            if node_details:
                d.update(node_details)
                results.append(d)
        return results

    def find_nodes_by_name(self, name: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE name = ?", (name,))
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d['properties'] = json.loads(d['properties']) if d['properties'] else {}
            results.append(d)
        return results

    def get_nodes_by_file(self, file_path: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE file_path = ?", (file_path,))
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d['properties'] = json.loads(d['properties']) if d['properties'] else {}
            results.append(d)
        return results

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM nodes WHERE type = ?", (node_type,))
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d['properties'] = json.loads(d['properties']) if d['properties'] else {}
            results.append(d)
        return results

    def search_nodes_keyword(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Performs exact keyword search using SQLite."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        like_query = f"%{query}%"
        cursor.execute("""
            SELECT * FROM nodes
            WHERE name LIKE ? OR properties LIKE ?
            LIMIT ?
        """, (like_query, like_query, limit))
        
        results = []
        for row in cursor.fetchall():
            d = dict(row)
            d['properties'] = json.loads(d['properties']) if d['properties'] else {}
            d['match_type'] = 'keyword'
            results.append(d)
        return results

    def resolve_unknown_edges(self):
        """Resolves edges with UNKNOWN targets to actual nodes."""
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute("SELECT source_id, target_id, type, properties FROM edges WHERE target_id LIKE 'UNKNOWN:%'")
        unknown_edges = cursor.fetchall()
        
        for edge in unknown_edges:
            source_id, target_id, edge_type, props_json = edge
            try:
                name = target_id.split(":", 1)[1]
            except IndexError:
                continue
            
            props = json.loads(props_json) if props_json else {}
            module_hint = props.get('module_hint')
            
            nodes = self.find_nodes_by_name(name)
            if nodes:
                candidates = nodes
                if module_hint:
                    # Normalize module hint (replace dots with slashes for path matching)
                    # e.g. "app.services.nse_service" -> "app/services/nse_service"
                    hint_path = module_hint.replace(".", "/")
                    
                    filtered = [n for n in nodes if hint_path in n['file_path']]
                    if filtered:
                        candidates = filtered
                
                for node in candidates:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO edges (source_id, target_id, type, properties)
                            VALUES (?, ?, ?, ?)
                        """, (source_id, node['id'], edge_type, props_json))
                    except Exception:
                        pass
                
                cursor.execute("DELETE FROM edges WHERE source_id=? AND target_id=? AND type=?", (source_id, target_id, edge_type))
        
        conn.commit()

    def close(self):
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
