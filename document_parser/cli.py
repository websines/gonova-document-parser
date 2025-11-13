"""Command-line interface for document processing."""

import json
import sys
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from document_parser.config import AccuracyMode, InferenceMode, settings
from document_parser.hybrid_processor import HybridDocumentProcessor

app = typer.Typer(
    name="docparse",
    help="Hybrid document processing with DeepSeek, Nanonets, and Granite",
    add_completion=False,
)
console = Console()


@app.command()
def process(
    pdf_path: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        dir_okay=False,
        help="Path to PDF file",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: <input>_processed.<format>)",
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, json, or html",
    ),
    accuracy_mode: str = typer.Option(
        settings.default_accuracy_mode.value,
        "--accuracy",
        "-a",
        help="Accuracy mode: fast, balanced, or maximum",
    ),
    inference_mode: str = typer.Option(
        settings.inference_mode.value,
        "--inference",
        "-i",
        help="Inference mode: transformers or vllm",
    ),
    vqa_questions: Optional[List[str]] = typer.Option(
        None,
        "--question",
        "-q",
        help="VQA questions (can be specified multiple times)",
    ),
    extract_signatures: bool = typer.Option(
        False,
        "--signatures",
        "-s",
        help="Enable signature detection",
    ),
    no_enrichment: bool = typer.Option(
        False,
        "--no-enrichment",
        help="Disable Granite semantic enrichment",
    ),
    show_graph: bool = typer.Option(
        False,
        "--show-graph",
        help="Display graph statistics",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose logging",
    ),
):
    """
    Process a PDF document with hybrid approach.

    Examples:

        # Basic processing with balanced accuracy
        docparse process document.pdf

        # Fast processing, markdown output
        docparse process document.pdf -a fast -f markdown

        # Maximum accuracy with VQA
        docparse process contract.pdf -a maximum -q "Who signed this?" -q "What is the date?"

        # Signature detection enabled
        docparse process legal_doc.pdf --signatures

        # vLLM server mode (faster for batch processing)
        docparse process document.pdf -i vllm
    """
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")

    # Show header
    console.print(
        Panel.fit(
            f"[bold cyan]Hybrid Document Processor[/bold cyan]\n"
            f"Accuracy: [yellow]{accuracy_mode}[/yellow] | "
            f"Inference: [yellow]{inference_mode}[/yellow] | "
            f"Enrichment: [yellow]{'disabled' if no_enrichment else 'enabled'}[/yellow]",
            border_style="cyan",
        )
    )

    try:
        # Initialize processor
        processor = HybridDocumentProcessor(
            accuracy_mode=AccuracyMode(accuracy_mode),
            inference_mode=InferenceMode(inference_mode),
            enable_enrichment=not no_enrichment,
        )

        # Process document
        console.print(f"\n[bold]Processing:[/bold] {pdf_path.name}")
        result = processor.process(
            pdf_path=pdf_path,
            output_format=output_format,
            vqa_questions=vqa_questions,
            extract_signatures=extract_signatures if extract_signatures else None,
        )

        # Determine output path
        if output is None:
            ext = {"markdown": "md", "json": "json", "html": "html"}[output_format]
            output = pdf_path.parent / f"{pdf_path.stem}_processed.{ext}"

        # Save output
        if output_format == "json":
            # Save as JSON with nodes/edges
            output_data = {
                "document_id": result.document_id,
                "filename": result.filename,
                "nodes": result.nodes,
                "edges": result.edges,
                "metadata": result.metadata,
                "vqa_answers": result.vqa_answers,
            }
            output.write_text(json.dumps(output_data, indent=2))
        else:
            # Save markdown/html content
            content = result.metadata.get("output", "")
            output.write_text(content)

        # Display results
        console.print(f"\n[green]✓[/green] Processing complete!")
        console.print(f"[bold]Output:[/bold] {output}")

        # Stats table
        stats = Table(title="Processing Statistics", show_header=True)
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", style="yellow")

        stats.add_row("Pages", str(result.metadata.get("num_pages", 0)))
        stats.add_row(
            "Processing Time", f"{result.metadata.get('processing_time', 0):.1f}s"
        )
        stats.add_row("Primary Processor", result.metadata["routing_info"]["primary_processor"])
        stats.add_row("Accuracy Mode", result.metadata["accuracy_mode"])

        if result.metadata.get("signatures_found"):
            stats.add_row("Signatures Found", str(len(result.metadata["signatures_found"])))

        if result.vqa_answers:
            stats.add_row("VQA Answers", str(len(result.vqa_answers)))

        console.print(stats)

        # Graph stats
        if show_graph:
            graph_table = Table(title="Graph Structure", show_header=True)
            graph_table.add_column("Type", style="cyan")
            graph_table.add_column("Count", style="yellow")

            graph_table.add_row("Nodes", str(len(result.nodes)))
            graph_table.add_row("Edges", str(len(result.edges)))

            # Node type breakdown
            node_types = {}
            for node in result.nodes:
                node_type = node.get("type", "unknown")
                node_types[node_type] = node_types.get(node_type, 0) + 1

            for node_type, count in sorted(node_types.items()):
                graph_table.add_row(f"  └─ {node_type}", str(count))

            console.print(graph_table)

        # VQA answers
        if result.vqa_answers:
            console.print("\n[bold cyan]VQA Answers:[/bold cyan]")
            for question, answer in result.vqa_answers.items():
                console.print(f"  [yellow]Q:[/yellow] {question}")
                console.print(f"  [green]A:[/green] {answer}\n")

        # Routing explanation
        if verbose:
            console.print("\n[bold cyan]Routing Decision:[/bold cyan]")
            routing = result.metadata["routing_info"]
            console.print(f"  Processor: {routing['primary_processor']}")
            console.print(f"  Reasoning: {routing['reasoning']}")
            console.print(f"  Enrichment: {routing['enrichment_recommended']}")

    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing PDF files",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: <input_dir>_processed)",
    ),
    pattern: str = typer.Option(
        "*.pdf",
        "--pattern",
        "-p",
        help="File pattern to match",
    ),
    output_format: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: markdown, json, or html",
    ),
    accuracy_mode: str = typer.Option(
        settings.default_accuracy_mode.value,
        "--accuracy",
        "-a",
        help="Accuracy mode: fast, balanced, or maximum",
    ),
    inference_mode: str = typer.Option(
        settings.inference_mode.value,
        "--inference",
        "-i",
        help="Inference mode: transformers or vllm",
    ),
    max_files: Optional[int] = typer.Option(
        None,
        "--max-files",
        "-n",
        help="Maximum number of files to process",
    ),
    continue_on_error: bool = typer.Option(
        True,
        "--continue-on-error",
        help="Continue processing if a file fails",
    ),
):
    """
    Process multiple PDF files in batch mode.

    Examples:

        # Process all PDFs in a directory
        docparse batch ./documents/

        # Process with maximum accuracy
        docparse batch ./contracts/ -a maximum

        # Process first 10 files with vLLM
        docparse batch ./invoices/ -i vllm -n 10

        # Custom output directory
        docparse batch ./input/ -o ./output/
    """
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Determine output directory
    if output_dir is None:
        output_dir = input_dir.parent / f"{input_dir.name}_processed"

    output_dir.mkdir(exist_ok=True, parents=True)

    # Find PDF files
    pdf_files = list(input_dir.glob(pattern))
    if max_files:
        pdf_files = pdf_files[:max_files]

    if not pdf_files:
        console.print(f"[yellow]No PDF files found matching pattern: {pattern}[/yellow]")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold cyan]Batch Processing[/bold cyan]\n"
            f"Files: [yellow]{len(pdf_files)}[/yellow] | "
            f"Accuracy: [yellow]{accuracy_mode}[/yellow] | "
            f"Inference: [yellow]{inference_mode}[/yellow]",
            border_style="cyan",
        )
    )

    # Initialize processor
    processor = HybridDocumentProcessor(
        accuracy_mode=AccuracyMode(accuracy_mode),
        inference_mode=InferenceMode(inference_mode),
    )

    # Process files
    successful = 0
    failed = 0
    total_pages = 0
    total_time = 0.0

    for i, pdf_path in enumerate(pdf_files, 1):
        console.print(f"\n[bold][{i}/{len(pdf_files)}][/bold] Processing: {pdf_path.name}")

        try:
            result = processor.process(
                pdf_path=pdf_path,
                output_format=output_format,
            )

            # Save output
            ext = {"markdown": "md", "json": "json", "html": "html"}[output_format]
            output_path = output_dir / f"{pdf_path.stem}.{ext}"

            if output_format == "json":
                output_data = {
                    "document_id": result.document_id,
                    "filename": result.filename,
                    "nodes": result.nodes,
                    "edges": result.edges,
                    "metadata": result.metadata,
                }
                output_path.write_text(json.dumps(output_data, indent=2))
            else:
                content = result.metadata.get("output", "")
                output_path.write_text(content)

            pages = result.metadata.get("num_pages", 0)
            time_elapsed = result.metadata.get("processing_time", 0)

            console.print(
                f"  [green]✓[/green] {pages} pages in {time_elapsed:.1f}s "
                f"({pages/time_elapsed:.1f} pages/sec)"
            )

            successful += 1
            total_pages += pages
            total_time += time_elapsed

        except Exception as e:
            console.print(f"  [red]✗[/red] Failed: {e}")
            failed += 1

            if not continue_on_error:
                raise typer.Exit(code=1)

    # Summary
    console.print("\n" + "=" * 60)
    console.print(f"[bold cyan]Batch Processing Complete[/bold cyan]")
    console.print(f"  Successful: [green]{successful}[/green]")
    console.print(f"  Failed: [red]{failed}[/red]")
    console.print(f"  Total Pages: {total_pages}")
    console.print(f"  Total Time: {total_time:.1f}s")
    console.print(f"  Average Speed: {total_pages/total_time:.1f} pages/sec")
    console.print(f"  Output Directory: {output_dir}")
    console.print("=" * 60)


@app.command()
def status():
    """Show processor status and configuration."""
    processor = HybridDocumentProcessor()
    status_info = processor.get_status()

    console.print(
        Panel.fit(
            f"[bold cyan]Document Processor Status[/bold cyan]",
            border_style="cyan",
        )
    )

    # Configuration
    config_table = Table(title="Configuration", show_header=True)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="yellow")

    config_table.add_row("Accuracy Mode", status_info["accuracy_mode"])
    config_table.add_row("Inference Mode", status_info["inference_mode"])
    config_table.add_row("Enrichment", str(status_info["enable_enrichment"]))
    config_table.add_row("Device", settings.torch_device)
    config_table.add_row("VRAM Limit", f"{settings.vram_limit_gb}GB")

    console.print(config_table)

    # Processors
    proc_table = Table(title="Processors", show_header=True)
    proc_table.add_column("Processor", style="cyan")
    proc_table.add_column("Status", style="yellow")
    proc_table.add_column("Model", style="green")

    processors = status_info["processors_loaded"]
    proc_table.add_row(
        "DeepSeek-OCR",
        "✓ Loaded" if processors["deepseek"] else "○ Not loaded",
        settings.deepseek_model,
    )
    proc_table.add_row(
        "Nanonets-OCR2-3B",
        "✓ Loaded" if processors["nanonets"] else "○ Not loaded",
        settings.nanonets_model,
    )
    proc_table.add_row(
        "Granite-Docling",
        "✓ Loaded" if processors["granite"] else "○ Not loaded",
        settings.granite_model,
    )

    console.print(proc_table)

    # vLLM endpoints
    if settings.inference_mode == InferenceMode.VLLM:
        vllm_table = Table(title="vLLM Endpoints", show_header=True)
        vllm_table.add_column("Service", style="cyan")
        vllm_table.add_column("URL", style="yellow")

        if settings.vllm_deepseek_url:
            vllm_table.add_row("DeepSeek", settings.vllm_deepseek_url)
        if settings.vllm_nanonets_url:
            vllm_table.add_row("Nanonets", settings.vllm_nanonets_url)
        if settings.vllm_granite_url:
            vllm_table.add_row("Granite", settings.vllm_granite_url)

        console.print(vllm_table)


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
