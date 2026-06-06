"""
Configuration management for the paper extraction pipeline.

Uses pydantic-settings to load and validate configuration from .env files.
This ensures type safety and provides clear error messages for misconfiguration.
"""

from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineSettings(BaseSettings):
    """
    Pipeline configuration loaded from .env file.
    
    Pydantic automatically:
    - Loads values from environment variables or .env file
    - Validates types (int, bool, Path, etc.)
    - Provides defaults
    - Raises clear errors if required fields are missing
    """
    
    # === LLM Configuration ===
    llm_model: str = Field(
        default="Qwen/Qwen3.5-35B-A3B",
        description="HuggingFace model ID for extraction"
    )
    
    # === GROBID Configuration ===
    grobid_url: str = Field(
        default="http://localhost:8070",
        description="GROBID server URL for PDF parsing"
    )
    
    # === Text Chunking Strategy ===
    chunk_strategy: Literal["section", "paragraph"] = Field(
        default="section",
        description="How to split text: by paper sections or paragraphs"
    )
    
    chunk_size: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Maximum tokens per chunk"
    )
    
    chunk_overlap: int = Field(
        default=100,
        ge=0,
        description="Token overlap between chunks for context preservation"
    )
    
    # === Pipeline Paths ===
    input_dir: Path = Field(
        default=Path("./inputs"),
        description="Directory containing input PDFs"
    )
    
    output_dir: Path = Field(
        default=Path("./results"),
        description="Directory for extracted JSON outputs"
    )

    tei_xml_dir: Path = Field(
        default=Path("./tei_xml"),
        description="Directory for saved intermediate TEI XML files"
    )
    
    log_dir: Path = Field(
        default=Path("./logs"),
        description="Directory for log files"
    )
    
    # === Processing Options ===
    batch_mode: bool = Field(
        default=True,
        description="Enable batch processing of multiple PDFs"
    )
    
    parallel_jobs: int = Field(
        default=1,
        ge=1,
        description="Number of PDFs to process in parallel"
    )
    
    enable_checkpoint: bool = Field(
        default=True,
        description="Enable checkpoint/resume capability"
    )
    
    checkpoint_file: Path = Field(
        default=Path("./checkpoint.json"),
        description="Checkpoint file location"
    )
    
    # === Logging Configuration ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    log_format: Literal["json", "text"] = Field(
        default="text",
        description="Log output format"
    )
    
    per_pdf_logs: bool = Field(
        default=True,
        description="Enable per-PDF log files"
    )
    
    # === LLM Extraction Settings ===
    llm_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts for failed LLM calls"
    )
    
    llm_timeout: int = Field(
        default=120,
        ge=10,
        description="Timeout for LLM calls in seconds"
    )
    
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0=deterministic, 1.0=creative)"
    )

    vllm_gpu_memory: float = Field(
        default=0.90,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory vLLM can use"
    )

    vllm_max_model_len: Optional[int] = Field(
        default=None,
        ge=256,
        description="Optional max model sequence length for vLLM"
    )

    vllm_max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum tokens generated per vLLM call"
    )
    
    # === Extraction Schema ===
    extraction_schema: str = Field(
        default="polymer-lccc",
        description="Target extraction format"
    )
    
    # === Development/Debug Options ===
    max_pdfs: int = Field(
        default=0,
        ge=0,
        description="Limit number of PDFs to process (0=no limit)"
    )
    
    save_intermediate: bool = Field(
        default=False,
        description="Save intermediate outputs (TEI XML, chunks)"
    )
    
    debug_mode: bool = Field(
        default=False,
        description="Enable verbose debugging output"
    )
    
    # Pydantic configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore extra env vars
    )
    
    @field_validator("input_dir", "output_dir", "log_dir", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Convert string paths to Path objects and resolve them."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    def __init__(self, **kwargs):
        """Initialize and perform cross-field validations."""
        super().__init__(**kwargs)
        
        # Validate chunk_overlap < chunk_size
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tei_xml_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def display_config(self) -> str:
        """Return a human-readable configuration summary."""
        config_lines = [
            "=" * 50,
            "Pipeline Configuration",
            "=" * 50,
            "",
            "LLM Settings:",
            f"  Model: {self.llm_model}",
            f"  vLLM GPU Memory: {self.vllm_gpu_memory}",
            f"  vLLM Max Tokens: {self.vllm_max_tokens}",
            f"  Temperature: {self.llm_temperature}",
            "",
            "Chunking Strategy:",
            f"  Strategy: {self.chunk_strategy}",
            f"  Chunk Size: {self.chunk_size} tokens",
            f"  Overlap: {self.chunk_overlap} tokens",
            "",
            "Paths:",
            f"  Input: {self.input_dir}",
            f"  Output: {self.output_dir}",
            f"  TEI XML: {self.tei_xml_dir}",
            f"  Logs: {self.log_dir}",
            "",
            "Processing:",
            f"  Batch Mode: {self.batch_mode}",
            f"  Parallel Jobs: {self.parallel_jobs}",
            f"  Checkpointing: {self.enable_checkpoint}",
            "",
            "Logging:",
            f"  Level: {self.log_level}",
            f"  JSON Logs: {self.json_logs}",
            f"  Per-PDF Logs: {self.per_pdf_logs}",
            "",
            "=" * 50,
        ]
        return "\n".join(config_lines)


# Global settings instance
settings = PipelineSettings()
