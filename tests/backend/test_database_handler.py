import pytest
from unittest.mock import MagicMock
from video_similarity_search.backend.database_handler import MilvusHandler

def test_milvus_handler_initialization(mock_milvus_client):
    """Test MilvusHandler initialization with passed parameters."""
    handler = MilvusHandler(
        collection_name="test_collection",
        reset_dataset=True,
        embedding_size=512,
        uri="http://test:19530",
        token="test:token"
    )
    
    mock_milvus_client.assert_called_with(uri="http://test:19530", token="test:token")
    assert handler.client.has_collection.called

def test_milvus_handler_search(mock_milvus_client):
    """Test search functionality."""
    # Configure mock to say collection does not exist, so we don't hit the RuntimeError
    mock_client_instance = mock_milvus_client.return_value
    mock_client_instance.has_collection.return_value = False
    
    handler = MilvusHandler(
        collection_name="test_collection",
        reset_dataset=False,
        embedding_size=512,
        uri="http://test:19530",
        token="test:token"
    )
    
    query_embedding = [0.1] * 512
    handler.search(query_embedding)
    
    handler.client.search.assert_called()
