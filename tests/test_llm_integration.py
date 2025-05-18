import asyncio
import types
import yaml
import pytest

import analysis.tools.molecules.llm_integration as llm

class DummyModel:
    def __init__(self, model_name, device="cpu"):
        self.args = (model_name, device)
    def encode(self, texts, batch_size=32):
        return [[float(len(t))] for t in texts]

def setup_dummy_env(monkeypatch, tmp_path):
    class DummyCollection:
        def __init__(self):
            self.docs = []
        def add(self, embeddings, documents, metadatas, ids):
            for e, d, m, i in zip(embeddings, documents, metadatas, ids):
                self.docs.append((e, d, m, i))
        def query(self, query_embeddings, n_results=5):
            documents = [d for _, d, _, _ in self.docs][:n_results]
            metadatas = [m for _, _, m, _ in self.docs][:n_results]
            ids = [i for _, _, _, i in self.docs][:n_results]
            return {"documents": [documents], "metadatas": [metadatas], "ids": [ids]}

    class DummyClient:
        def __init__(self, path=None):
            self.path = path
        def get_or_create_collection(self, name, metadata=None):
            return DummyCollection()

    dummy_chroma = types.SimpleNamespace(PersistentClient=lambda path=None: DummyClient())
    monkeypatch.setattr(llm, "chromadb", dummy_chroma)
    monkeypatch.setattr(llm, "SentenceTransformer", DummyModel)

    async def dummy_completion(client, prompt, max_tokens=None, temperature=0.7, system_prompt=None):
        return f"answer:{prompt}"
    monkeypatch.setattr(llm, "create_completion", dummy_completion)
    monkeypatch.setattr(llm, "ClaudeClient", lambda cfg: "client")

    cfg = {
        "embeddings": {"model": "dummy"},
        "vector_db": {"collection_name": "test", "persist_directory": str(tmp_path)},
    }
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.dump(cfg))
    return llm.LocalKnowledgeRAG(str(cfg_file))


def test_initialize_vector_db(monkeypatch, tmp_path):
    rag = setup_dummy_env(monkeypatch, tmp_path)
    assert rag.collection is not None


def test_generate_embeddings(monkeypatch, tmp_path):
    rag = setup_dummy_env(monkeypatch, tmp_path)
    embs = rag._generate_embeddings(["hi", "there"])
    assert embs == [[2.0], [5.0]]


@pytest.mark.asyncio
async def test_rag_generation(monkeypatch, tmp_path):
    rag = setup_dummy_env(monkeypatch, tmp_path)
    rag.collection.add([[1.0]], ["doc"], [{"id": 1}], ids=["1"])
    result = await rag.retrieval_augmented_generation("q")
    assert result.startswith("answer:q")
