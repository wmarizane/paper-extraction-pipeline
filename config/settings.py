"""
Configuration management for the paper extraction pipeline.

Uses pydantic-settings to load and validate configuration from .env files.
This ensures type safety and provides clear error messages for misconfiguration.
"""

from pathlib import Path
from typing import Literal

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
        default="qwen3:4b",
        description="Ollama model name for extraction"
    )
    
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
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
    
    json_logs: bool = Field(
        default=True,
        description="Enable JSON log output for automated analysis"
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
        extra="ignore"  # Ignore extra fields in .env
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        # Note: chunk_size not available yet during validation
        # Will be validated at runtime in __init__
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
            f"  Ollama URL: {self.ollama_url}",
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


# Global settings instance - import this in other modules
settings = PipelineSettings()


if __name__ == "__main__":
    """Test configuration loading."""
    print("Testing configuration loading...\n")
    
    try:
        test_settings = PipelineSettings()
        print(test_settings.display_config())
        print("\nConfiguration loaded successfully!")
        
        # Show which .env file was used (if any)
        env_file = Path(".env")
        if env_file.exists():
            print(f"\nUsing configuration from: {env_file.absolute()}")
        else:
            print("\n No .env file found, using defaults")
            print("   Copy .env.example to .env to customize settings")
            
    except Exception as e:
        print(f"\nConfiguration error: {e}")
        print("\nCheck your .env file and ensure all required settings are valid.")
