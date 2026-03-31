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
        default="Qwen/Qwen2.5-27B-Instruct",
        description="HuggingFace model ID for vLLM extraction"
    )
    
    # vLLM-specific settings
    vllm_gpu_memory: float = Field(
        default=0.9,
        ge=0.1,
        le=1.0,
        description="Fraction of GPU memory to use (0.1-1.0)"
    )
    
    vllm_max_tokens: int = Field(
        default=2048,
        ge=128,
        le=8192,
        description="Maximum tokens to generate per response"
    )
    
    vllm_max_model_len: Optional[int] = Field(
        default=None,
        description="Maximum sequence length (None = auto-detect from model)"
    )
    
    # LLM generation settings
    llm_temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0=deterministic, higher=creative)"
    )
    
    llm_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts on LLM failure"
    )
    
    llm_timeout: int = Field(
        default=300,
        ge=30,
        description="LLM call timeout in seconds"
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
    
    # === Model Configuration Class ===
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


# Global settings instance
settings = PipelineSettings()
