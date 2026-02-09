import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from video_similarity_search.backend.video_processor import VideoProcessor
from PIL import Image

def test_extract_frame_embeddings_batching(mock_model):
    """Test that VideoProcessor correctly batches frames."""
    processor = VideoProcessor(mock_model)
    
    # Mock cv2 and video capture
    with patch("cv2.VideoCapture") as mock_cap_cls, \
         patch("pathlib.Path.exists", return_value=True):
        mock_cap = mock_cap_cls.return_value
        mock_cap.isOpened.side_effect = [True, True, True, True, True, False] # 5 frames, loop until False
        
        # Create dummy frames
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, dummy_frame)] * 5 + [(False, None)]
        
        # Mock model output
        mock_model.extract_image_features.return_value = np.zeros((2, 512)) # Batch of 2 (since frame_skip=2, 5 frames -> indices 0, 2, 4. Batch size 2 -> First batch [0, 2], Second batch [4])
        # Wait, if frame_skip=2:
        # Frame 0: Process (batch: [0])
        # Frame 1: Skip
        # Frame 2: Process (batch: [0, 2]) -> Batch full (size=2) -> Process -> batch=[]
        # Frame 3: Skip
        # Frame 4: Process (batch: [4]) -> End -> Process remaining
        
        embeddings, indices = processor.extract_frame_embeddings("dummy.mp4", frame_skip=2, batch_size=2)
        
        assert len(indices) == 3 # Frames 0, 2, 4
        assert mock_model.extract_image_features.call_count == 2 # 1 full batch + 1 remaining batch
        
def test_extract_frame_embeddings_empty_video(mock_model):
    """Test that VideoProcessor handles empty videos correctly."""
    processor = VideoProcessor(mock_model)
    
    with patch("cv2.VideoCapture") as mock_cap_cls, \
         patch("pathlib.Path.exists", return_value=True):
        mock_cap = mock_cap_cls.return_value
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        
        embeddings, indices = processor.extract_frame_embeddings("dummy.mp4")
        
        assert len(indices) == 0
        assert embeddings.shape == (0, 512)
