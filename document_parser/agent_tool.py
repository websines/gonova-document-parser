"""Agent toolkit integration for document processing.

This module provides a tool interface compatible with LangChain, AutoGen,
CrewAI, and other agent frameworks. Agents can call this tool to process
documents and get structured outputs for RAG pipelines.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from document_parser.config import AccuracyMode, InferenceMode
from document_parser.hybrid_processor import GraphDocument, HybridDocumentProcessor


class DocumentProcessingToolInput(BaseModel):
    """Input schema for document processing tool."""

    pdf_path: str = Field(
        ...,
        description="Absolute path to the PDF file to process",
    )
    accuracy_mode: str = Field(
        default="balanced",
        description="Processing accuracy: 'fast', 'balanced', or 'maximum'",
    )
    vqa_questions: Optional[List[str]] = Field(
        default=None,
        description="Optional list of questions to answer about the document",
    )
    extract_signatures: Optional[bool] = Field(
        default=None,
        description="Whether to detect and extract signatures (default: auto-detect)",
    )
    output_format: str = Field(
        default="markdown",
        description="Output format: 'markdown', 'json', or 'html'",
    )


class DocumentProcessingToolOutput(BaseModel):
    """Output schema for document processing tool."""

    success: bool = Field(..., description="Whether processing succeeded")
    document_id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    content: str = Field(
        ..., description="Processed document content in requested format"
    )
    nodes: List[Dict] = Field(
        ..., description="Graph nodes (sections, tables, paragraphs)"
    )
    edges: List[Dict] = Field(..., description="Graph edges (relationships)")
    metadata: Dict[str, Any] = Field(..., description="Processing metadata and stats")
    vqa_answers: Optional[Dict[str, str]] = Field(
        default=None, description="Answers to VQA questions if provided"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class DocumentProcessingTool:
    """
    Document processing tool for agent frameworks.

    Compatible with:
    - LangChain (as a StructuredTool)
    - AutoGen (as a function tool)
    - CrewAI (as a BaseTool)
    - Custom agent frameworks

    Example usage with LangChain:
        ```python
        from langchain.agents import StructuredTool

        tool = DocumentProcessingTool()
        langchain_tool = StructuredTool.from_function(
            func=tool.run,
            name=tool.name,
            description=tool.description,
            args_schema=DocumentProcessingToolInput,
        )
        ```

    Example usage with AutoGen:
        ```python
        tool = DocumentProcessingTool()
        autogen_tool = {
            "name": tool.name,
            "description": tool.description,
            "function": tool.run,
        }
        ```
    """

    name = "process_compliance_document"
    description = """
Process financial, legal, or compliance PDF documents with high accuracy.

This tool uses a hybrid approach with multiple specialized models:
- DeepSeek-OCR: Fast processing for standard typed content and tables
- Nanonets-OCR2-3B: Handwritten content, signatures, and VQA
- Granite-Docling: Semantic structure enrichment

Returns structured output ready for graph-vector database ingestion with:
- Nodes: Sections, tables, paragraphs, and other document elements
- Edges: Relationships between elements (follows, contains, references)
- Metadata: Processing stats, detected features, routing decisions
- VQA answers: If questions were provided

Best for:
- Financial reports with complex tables
- Legal documents with signatures
- Compliance forms with handwritten sections
- Mixed content documents (typed + handwritten)

Input format:
{
    "pdf_path": "/path/to/document.pdf",
    "accuracy_mode": "balanced",  # fast, balanced, or maximum
    "vqa_questions": ["What is total revenue?", "Who signed this?"],
    "extract_signatures": true,
    "output_format": "markdown"  # markdown, json, or html
}
"""

    def __init__(
        self,
        inference_mode: InferenceMode = InferenceMode.TRANSFORMERS,
        enable_enrichment: bool = True,
    ):
        """
        Initialize document processing tool.

        Args:
            inference_mode: transformers (direct) or vllm (server-based)
            enable_enrichment: Enable Granite semantic enrichment
        """
        self.processor = HybridDocumentProcessor(
            inference_mode=inference_mode,
            enable_enrichment=enable_enrichment,
        )

    def run(
        self,
        pdf_path: str,
        accuracy_mode: str = "balanced",
        vqa_questions: Optional[List[str]] = None,
        extract_signatures: Optional[bool] = None,
        output_format: str = "markdown",
    ) -> DocumentProcessingToolOutput:
        """
        Process a PDF document and return structured output.

        Args:
            pdf_path: Absolute path to PDF file
            accuracy_mode: Processing accuracy (fast, balanced, maximum)
            vqa_questions: Optional questions to answer
            extract_signatures: Whether to detect signatures
            output_format: Output format (markdown, json, html)

        Returns:
            DocumentProcessingToolOutput with processed document
        """
        try:
            # Validate input
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                return DocumentProcessingToolOutput(
                    success=False,
                    document_id="",
                    filename=pdf_path.name,
                    content="",
                    nodes=[],
                    edges=[],
                    metadata={},
                    error=f"File not found: {pdf_path}",
                )

            # Process document
            result: GraphDocument = self.processor.process(
                pdf_path=pdf_path,
                output_format=output_format,
                accuracy_mode=AccuracyMode(accuracy_mode),
                vqa_questions=vqa_questions,
                extract_signatures=extract_signatures,
            )

            # Extract content based on format
            if output_format == "markdown":
                content = result.metadata.get("output", "")
            elif output_format == "json":
                import json

                content = json.dumps(
                    {
                        "nodes": result.nodes,
                        "edges": result.edges,
                        "metadata": result.metadata,
                    },
                    indent=2,
                )
            else:
                content = result.metadata.get("output", "")

            return DocumentProcessingToolOutput(
                success=True,
                document_id=result.document_id,
                filename=result.filename,
                content=content,
                nodes=result.nodes,
                edges=result.edges,
                metadata=result.metadata,
                vqa_answers=result.vqa_answers,
            )

        except Exception as e:
            return DocumentProcessingToolOutput(
                success=False,
                document_id="",
                filename=Path(pdf_path).name,
                content="",
                nodes=[],
                edges=[],
                metadata={},
                error=str(e),
            )

    def cleanup(self):
        """Cleanup and unload models."""
        self.processor.cleanup()


# LangChain integration helper
def create_langchain_tool(
    inference_mode: InferenceMode = InferenceMode.TRANSFORMERS,
    enable_enrichment: bool = True,
):
    """
    Create LangChain-compatible tool.

    Args:
        inference_mode: transformers or vllm
        enable_enrichment: Enable Granite enrichment

    Returns:
        LangChain StructuredTool instance

    Example:
        ```python
        from langchain.agents import AgentExecutor, create_openai_functions_agent
        from langchain_openai import ChatOpenAI

        tool = create_langchain_tool()
        llm = ChatOpenAI(model="gpt-4")
        agent = create_openai_functions_agent(llm, [tool], prompt)
        executor = AgentExecutor(agent=agent, tools=[tool])

        result = executor.invoke({
            "input": "Process the financial report and tell me the revenue"
        })
        ```
    """
    try:
        from langchain.agents import StructuredTool

        tool_instance = DocumentProcessingTool(
            inference_mode=inference_mode,
            enable_enrichment=enable_enrichment,
        )

        return StructuredTool.from_function(
            func=tool_instance.run,
            name=tool_instance.name,
            description=tool_instance.description,
            args_schema=DocumentProcessingToolInput,
        )
    except ImportError:
        raise ImportError(
            "LangChain not installed. Install with: pip install langchain"
        )


# AutoGen integration helper
def create_autogen_tool(
    inference_mode: InferenceMode = InferenceMode.TRANSFORMERS,
    enable_enrichment: bool = True,
) -> Dict[str, Any]:
    """
    Create AutoGen-compatible tool.

    Args:
        inference_mode: transformers or vllm
        enable_enrichment: Enable Granite enrichment

    Returns:
        Dict with tool configuration

    Example:
        ```python
        from autogen import AssistantAgent, UserProxyAgent

        tool = create_autogen_tool()
        assistant = AssistantAgent(
            "assistant",
            llm_config={"functions": [tool]},
        )
        user_proxy = UserProxyAgent(
            "user_proxy",
            function_map={tool["name"]: tool["function"]},
        )
        ```
    """
    tool_instance = DocumentProcessingTool(
        inference_mode=inference_mode,
        enable_enrichment=enable_enrichment,
    )

    return {
        "name": tool_instance.name,
        "description": tool_instance.description,
        "function": tool_instance.run,
        "parameters": {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "Absolute path to the PDF file",
                },
                "accuracy_mode": {
                    "type": "string",
                    "enum": ["fast", "balanced", "maximum"],
                    "default": "balanced",
                },
                "vqa_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional questions to answer",
                },
                "extract_signatures": {
                    "type": "boolean",
                    "description": "Whether to detect signatures",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["markdown", "json", "html"],
                    "default": "markdown",
                },
            },
            "required": ["pdf_path"],
        },
    }
