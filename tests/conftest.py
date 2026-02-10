import pytest
from unittest.mock import MagicMock
from video_similarity_search.schema import AppConfig
from video_similarity_search.backend.model import VLMBaseModel

@pytest.fixture
def mock_milvus_client(mocker):
    return mocker.patch("video_similarity_search.backend.database_handler.MilvusClient")

@pytest.fixture
def mock_app_config():
    config = MagicMock(spec=AppConfig)
    config.collection_name = "test_collection"
    config.reset_dataset = True
    config.frame_skip = 2
    config.batch_size = 32
    config.milvus_uri = "http://localhost:19530"
    config.milvus_token = "root:Milvus"
    return config

@pytest.fixture
def mock_model():
    model = MagicMock(spec=VLMBaseModel)
    model.get_embedding_length.return_value = 512
    return model
