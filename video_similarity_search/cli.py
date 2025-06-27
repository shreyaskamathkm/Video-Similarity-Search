import logging
from pathlib import Path

import click
from cloudpathlib import AnyPath

from video_similarity_search.backend.database_handler import MilvusHandler, VideoDatabase
from video_similarity_search.backend.model import model_factory
from video_similarity_search.backend.query_result_formatter import QueryResultFormatter
from video_similarity_search.backend.search import VideoSearch
from video_similarity_search.backend.video_handler import VideoHandler
from video_similarity_search.backend.video_processor import VideoProcessor
from video_similarity_search.backend.video_segment_extractor import VideoSegmentExtractor
from video_similarity_search.schema import AppConfig

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """A command-line interface for video similarity search."""
    pass


@cli.command("run_video_similarity")
@click.option(
    "--config-path",
    type=AnyPath,
    help="Path to the config file.",
)
def run_video_similarity(config_path: Path | str) -> None:
    """Runs the video similarity search.

    Args:
        config_path: The path to the config file.
    """
    # Load configuration
    app_config = AppConfig.from_yaml(config_path)

    # Initialize core model
    model = model_factory(
        app_config.model_name, app_config.model_architecture, app_config.model_pretrained
    )

    # Initializes and wires up video processing components
    video_processor = VideoProcessor(model)
    video_segment_extractor = VideoSegmentExtractor()
    query_result_formatter = QueryResultFormatter(video_segment_extractor)
    video_handler = VideoHandler(video_processor, video_segment_extractor, query_result_formatter)

    # Initializes and wires up database components
    milvus_handler = MilvusHandler(
        collection_name=app_config.collection_name,
        reset_dataset=app_config.reset_dataset,
        embedding_size=model.get_embedding_length(),
    )
    video_database = VideoDatabase(
        model, video_handler, milvus_handler, frame_skip=app_config.frame_skip
    )

    # Populate the database with video embeddings
    logger.info(f"Populating database from folder: {app_config.video_folder}")
    video_database.add_videos_from_folder(app_config.video_folder)

    # Log information about the database content
    milvus_handler.get_all_videos_and_frame_indices()

    # Perform the search
    logger.info(f"Performing search for query: '{app_config.query}'")
    video_search = VideoSearch(model, milvus_handler)
    results = video_search.search_by_text(app_config.query)

    # Present the results
    logger.info("Displaying search results...")
    video_handler.present_query_results(results)


if __name__ == "__main__":
    cli()
