import os
import tempfile
import pytest
import graphite_db as graphite

def test_version():
    assert graphite.__version__ is not None
    assert len(graphite.__version__) > 0

def test_in_memory_basic_lifecycle():
    db = graphite.in_memory(dim=4)
    
    v1 = [1.0, 0.0, 0.0, 0.0]
    v2 = [0.95, 0.05, 0.0, 0.0]
    
    id1 = db.insert("AuthService", "Module", "Validates JWTs", v1)
    id2 = db.insert("JwtValidator", "Component", "Parses RS256 claims", v2)
    
    assert id1 >= 0
    assert id2 > id1
    
    db.add_edge(id1, id2, "USES", 0.95)
    
    stats = db.inspect()
    assert stats["nodes_count"] == 2
    assert stats["edges_count"] == 1
    assert stats["vector_dim"] == 4
    assert stats["is_in_memory"] is True
    
    # Query with vector
    q_vec = [0.98, 0.02, 0.0, 0.0]
    res = db.retrieve_context(q_vec, query_text="Auth", threshold=0.50)
    
    assert res.token_count > 0
    assert "AuthService" in res.markdown
    assert "JwtValidator" in res.markdown
    
    res_dict = res.to_dict()
    assert res_dict["token_count"] == res.token_count
    assert res_dict["markdown"] == res.markdown

def test_disk_backed_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.graph")
        
        # 1. Create and populate
        db = graphite.open(db_path, dim=4)
        id_a = db.insert("Alice", "Person", "Lead Architect", [1.0, 0.0, 0.0, 0.0])
        id_b = db.insert("ProjectTitan", "Project", "Core Engine", [0.9, 0.1, 0.0, 0.0])
        db.connect("Alice", "ProjectTitan", "LEADS", 0.99)
        db.flush()
        db.close()
        
        assert os.path.exists(db_path)
        
        # 2. Re-open and verify
        db2 = graphite.open(db_path, dim=4)
        stats = db2.inspect()
        assert stats["nodes_count"] == 2
        assert stats["edges_count"] == 1
        
        res = db2.retrieve_context([1.0, 0.0, 0.0, 0.0], threshold=0.50)
        assert "Alice" in res.markdown
        assert "ProjectTitan" in res.markdown
        db2.close()

def test_context_manager():
    with graphite.in_memory(dim=4) as db:
        id_node = db.insert("Server", "Infra", "Core Server", [0.5, 0.5, 0.0, 0.0])
        assert id_node >= 0
        stats = db.inspect()
        assert stats["nodes_count"] == 1

def test_embed_local_fastembed():
    vec = graphite.embed("Hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384
    
    batch = graphite.embed_batch(["First sentence", "Second sentence"])
    assert len(batch) == 2
    assert len(batch[0]) == 384
    assert len(batch[1]) == 384

def test_plain_text_query_with_fastembed():
    db = graphite.in_memory(dim=384)
    
    # Ingest direct text with auto embeddings
    db.ingest(text="# Person\nAlice is the principal software architect.", title="AliceProfile")
    db.ingest(text="# Project\nProject Apollo is a high-throughput microservices system.", title="ProjectApollo")
    
    # Query with natural language text
    result = db.query("Who is leading Project Apollo?", top_k=3, threshold=0.50)
    
    assert result.token_count > 0
    assert "Project Apollo" in result.markdown
