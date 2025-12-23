"""
Report generation utilities.

This module provides:
- Excel report generation
- PDF report generation  
- HTML report generation
- Report templates
"""

import numpy as np
from typing import List, Dict, Optional, Union
from pathlib import Path
from datetime import datetime
import json

from src.logger import get_logger

logger = get_logger(__name__)


class ExcelReportGenerator:
    """
    Generate Excel reports with statistics and charts.
    
    Uses openpyxl for Excel file generation.
    """
    
    def __init__(self):
        """Initialize ExcelReportGenerator."""
        try:
            import openpyxl
            from openpyxl.styles import Font, Alignment, PatternFill
            from openpyxl.chart import BarChart, PieChart, LineChart, Reference
            
            self.openpyxl = openpyxl
            self.Font = Font
            self.Alignment = Alignment
            self.PatternFill = PatternFill
            self.BarChart = BarChart
            self.PieChart = PieChart
            self.LineChart = LineChart
            self.Reference = Reference
            
        except ImportError:
            logger.error("openpyxl not installed. Install with: pip install openpyxl")
            raise
    
    def generate_defect_report(self, statistics: Dict, output_path: str):
        """
        Generate defect analysis Excel report.
        
        Args:
            statistics: Statistics dictionary from compute_batch_statistics
            output_path: Output Excel file path
        """
        logger.info(f"Generating Excel report: {output_path}")
        
        # Create workbook
        wb = self.openpyxl.Workbook()
        
        # Remove default sheet
        wb.remove(wb.active)
        
        # 1. Summary sheet
        self._create_summary_sheet(wb, statistics)
        
        # 2. Per-image statistics sheet
        self._create_per_image_sheet(wb, statistics)
        
        # 3. Size distribution sheet
        self._create_size_distribution_sheet(wb, statistics)
        
        # Save workbook
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(output_path)
        
        logger.info(f"Excel report saved to: {output_path}")
    
    def _create_summary_sheet(self, wb, statistics: Dict):
        """Create summary statistics sheet."""
        ws = wb.create_sheet("Summary")
        
        # Title
        ws['A1'] = "Defect Analysis Summary"
        ws['A1'].font = self.Font(size=16, bold=True)
        ws.merge_cells('A1:B1')
        
        # Date
        ws['A2'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws.merge_cells('A2:B2')
        
        # Statistics
        row = 4
        header_fill = self.PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = self.Font(color="FFFFFF", bold=True)
        
        # Header
        ws[f'A{row}'] = "Metric"
        ws[f'B{row}'] = "Value"
        ws[f'A{row}'].fill = header_fill
        ws[f'B{row}'].fill = header_fill
        ws[f'A{row}'].font = header_font
        ws[f'B{row}'].font = header_font
        
        row += 1
        
        # Data
        metrics = [
            ("Total Images", statistics.get('total_images', 0)),
            ("Images Processed", statistics.get('images_processed', 0)),
            ("Images with Defects", statistics.get('images_with_defects', 0)),
            ("Images without Defects", statistics.get('images_without_defects', 0)),
            ("Total Defects", statistics.get('total_defects', 0)),
            ("Mean Defects per Image", f"{statistics.get('mean_defects_per_image', 0):.2f}"),
            ("Mean Coverage Ratio", f"{statistics.get('mean_coverage_ratio', 0):.4f}"),
            ("Std Coverage Ratio", f"{statistics.get('std_coverage_ratio', 0):.4f}"),
        ]
        
        for metric, value in metrics:
            ws[f'A{row}'] = metric
            ws[f'B{row}'] = value
            row += 1
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 30
        ws.column_dimensions['B'].width = 20
    
    def _create_per_image_sheet(self, wb, statistics: Dict):
        """Create per-image statistics sheet."""
        ws = wb.create_sheet("Per-Image Stats")
        
        # Header
        headers = ["Image Name", "Num Defects", "Total Area", "Coverage Ratio", 
                  "Largest Defect", "Mean Defect Size"]
        
        header_fill = self.PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = self.Font(color="FFFFFF", bold=True)
        
        for col, header in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = self.Alignment(horizontal='center')
        
        # Data
        per_image_stats = statistics.get('per_image_stats', [])
        
        for row, stats in enumerate(per_image_stats, start=2):
            ws.cell(row=row, column=1, value=stats.get('image_name', ''))
            ws.cell(row=row, column=2, value=stats.get('num_defects', 0))
            ws.cell(row=row, column=3, value=stats.get('total_area', 0))
            ws.cell(row=row, column=4, value=f"{stats.get('coverage_ratio', 0):.4f}")
            ws.cell(row=row, column=5, value=stats.get('largest_defect', 0))
            ws.cell(row=row, column=6, value=f"{stats.get('mean_defect_size', 0):.2f}")
        
        # Adjust column widths
        for col in range(1, 7):
            ws.column_dimensions[chr(64 + col)].width = 18
    
    def _create_size_distribution_sheet(self, wb, statistics: Dict):
        """Create defect size distribution sheet."""
        ws = wb.create_sheet("Size Distribution")
        
        # Header
        ws['A1'] = "Bin Range"
        ws['B1'] = "Frequency"
        
        header_fill = self.PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = self.Font(color="FFFFFF", bold=True)
        ws['A1'].fill = header_fill
        ws['B1'].fill = header_fill
        ws['A1'].font = header_font
        ws['B1'].font = header_font
        
        # Data
        size_dist = statistics.get('defect_size_distribution', {})
        histogram = size_dist.get('histogram', [])
        bin_edges = size_dist.get('bin_edges', [])
        
        for i, (freq, bin_start) in enumerate(zip(histogram, bin_edges[:-1]), start=2):
            bin_end = bin_edges[i - 1]
            ws[f'A{i}'] = f"{bin_start:.0f} - {bin_end:.0f}"
            ws[f'B{i}'] = freq
        
        # Add chart
        if histogram:
            chart = self.BarChart()
            chart.title = "Defect Size Distribution"
            chart.x_axis.title = "Size Range (pixels)"
            chart.y_axis.title = "Frequency"
            
            data = self.Reference(ws, min_col=2, min_row=1, max_row=len(histogram) + 1)
            cats = self.Reference(ws, min_col=1, min_row=2, max_row=len(histogram) + 1)
            
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)
            
            ws.add_chart(chart, "D2")
        
        # Adjust column widths
        ws.column_dimensions['A'].width = 20
        ws.column_dimensions['B'].width = 15
    
    def generate_training_report(self, history: Dict, output_path: str):
        """
        Generate training history Excel report.
        
        Args:
            history: Training history dictionary
            output_path: Output Excel file path
        """
        logger.info(f"Generating training report: {output_path}")
        
        wb = self.openpyxl.Workbook()
        wb.remove(wb.active)
        
        # Create metrics sheet
        ws = wb.create_sheet("Training History")
        
        # Headers
        headers = ["Epoch"] + list(history.keys())
        header_fill = self.PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = self.Font(color="FFFFFF", bold=True)
        
        for col, header in enumerate(headers, start=1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.fill = header_fill
            cell.font = header_font
        
        # Data
        num_epochs = len(history[list(history.keys())[0]])
        
        for epoch in range(num_epochs):
            ws.cell(row=epoch + 2, column=1, value=epoch + 1)
            
            for col, metric in enumerate(history.keys(), start=2):
                value = history[metric][epoch]
                ws.cell(row=epoch + 2, column=col, value=value)
        
        # Add loss chart
        if 'train_loss' in history:
            chart = self.LineChart()
            chart.title = "Loss Curves"
            chart.x_axis.title = "Epoch"
            chart.y_axis.title = "Loss"
            
            train_loss_data = self.Reference(ws, min_col=2, min_row=1, max_row=num_epochs + 1)
            chart.add_data(train_loss_data, titles_from_data=True)
            
            if 'val_loss' in history:
                val_loss_col = list(history.keys()).index('val_loss') + 2
                val_loss_data = self.Reference(ws, min_col=val_loss_col, min_row=1, max_row=num_epochs + 1)
                chart.add_data(val_loss_data, titles_from_data=True)
            
            epochs_ref = self.Reference(ws, min_col=1, min_row=2, max_row=num_epochs + 1)
            chart.set_categories(epochs_ref)
            
            ws.add_chart(chart, f"A{num_epochs + 5}")
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wb.save(output_path)
        
        logger.info(f"Training report saved to: {output_path}")


class PDFReportGenerator:
    """
    Generate PDF reports with matplotlib figures.
    
    Uses matplotlib to save figures to PDF.
    """
    
    def __init__(self):
        """Initialize PDFReportGenerator."""
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            self.PdfPages = PdfPages
        except ImportError:
            logger.error("matplotlib not installed properly")
            raise
    
    def generate_defect_report(self, statistics: Dict, figures: List,
                              output_path: str):
        """
        Generate defect analysis PDF report.
        
        Args:
            statistics: Statistics dictionary
            figures: List of matplotlib figures
            output_path: Output PDF file path
        """
        logger.info(f"Generating PDF report: {output_path}")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with self.PdfPages(output_path) as pdf:
            # Add title page
            self._add_title_page(pdf, statistics)
            
            # Add all figures
            for fig in figures:
                pdf.savefig(fig, bbox_inches='tight')
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Defect Analysis Report'
            d['Author'] = 'Industrial Defect Segmentation System'
            d['Subject'] = 'Defect Statistics and Visualization'
            d['CreationDate'] = datetime.now()
        
        logger.info(f"PDF report saved to: {output_path}")
    
    def _add_title_page(self, pdf, statistics: Dict):
        """Add title page to PDF."""
        import matplotlib.pyplot as plt
        
        fig = plt.figure(figsize=(8.5, 11))
        fig.text(0.5, 0.7, 'Defect Analysis Report', 
                ha='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='center', fontsize=12)
        
        # Add summary statistics
        summary_text = f"""
        Total Images: {statistics.get('total_images', 0)}
        Images with Defects: {statistics.get('images_with_defects', 0)}
        Total Defects: {statistics.get('total_defects', 0)}
        Mean Defects per Image: {statistics.get('mean_defects_per_image', 0):.2f}
        Mean Coverage Ratio: {statistics.get('mean_coverage_ratio', 0):.4f}
        """
        
        fig.text(0.5, 0.4, summary_text, ha='center', fontsize=11,
                family='monospace')
        
        plt.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


class HTMLReportGenerator:
    """
    Generate HTML reports with embedded charts.
    
    Creates standalone HTML files with statistics and visualizations.
    """
    
    def __init__(self):
        """Initialize HTMLReportGenerator."""
        pass
    
    def generate_defect_report(self, statistics: Dict, chart_paths: Dict[str, str],
                              output_path: str):
        """
        Generate defect analysis HTML report.
        
        Args:
            statistics: Statistics dictionary
            chart_paths: Dictionary mapping chart names to file paths
            output_path: Output HTML file path
        """
        logger.info(f"Generating HTML report: {output_path}")
        
        html_content = self._create_html_template(statistics, chart_paths)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {output_path}")
    
    def _create_html_template(self, statistics: Dict, chart_paths: Dict[str, str]) -> str:
        """Create HTML report template."""
        
        # Header
        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Defect Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }
        .stat-label {
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .stat-value {
            font-size: 28px;
            font-weight: bold;
            color: #2c3e50;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Defect Analysis Report</h1>
        <p><strong>Generated:</strong> {timestamp}</p>
        
        <h2>📊 Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Images</div>
                <div class="stat-value">{total_images}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Images with Defects</div>
                <div class="stat-value">{images_with_defects}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Total Defects</div>
                <div class="stat-value">{total_defects}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Defects/Image</div>
                <div class="stat-value">{mean_defects:.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mean Coverage Ratio</div>
                <div class="stat-value">{mean_coverage:.4f}</div>
            </div>
        </div>
""".format(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_images=statistics.get('total_images', 0),
            images_with_defects=statistics.get('images_with_defects', 0),
            total_defects=statistics.get('total_defects', 0),
            mean_defects=statistics.get('mean_defects_per_image', 0),
            mean_coverage=statistics.get('mean_coverage_ratio', 0)
        )
        
        # Add charts
        html += "\n        <h2>📈 Visualizations</h2>\n"
        
        for chart_name, chart_path in chart_paths.items():
            # Use relative path
            rel_path = Path(chart_path).name
            html += f"""
        <div class="chart-container">
            <h3>{chart_name.replace('_', ' ').title()}</h3>
            <img src="{rel_path}" alt="{chart_name}">
        </div>
"""
        
        # Footer
        html += """
        <div class="footer">
            <p>Generated by Industrial Defect Segmentation System</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html


class ReportManager:
    """
    Manage report generation workflow.
    
    Coordinates statistics computation, visualization, and report generation.
    """
    
    def __init__(self):
        """Initialize ReportManager."""
        self.excel_generator = None
        self.pdf_generator = None
        self.html_generator = HTMLReportGenerator()
    
    def generate_complete_report(self, mask_paths: List[str],
                                output_dir: str,
                                report_formats: List[str] = ['excel', 'pdf', 'html']):
        """
        Generate complete report in multiple formats.
        
        Args:
            mask_paths: List of mask file paths
            output_dir: Output directory for reports
            report_formats: List of formats to generate ('excel', 'pdf', 'html')
        """
        from src.utils.statistics import DefectStatistics
        from src.utils.visualization import DefectVisualizer
        
        logger.info("Starting complete report generation...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Compute statistics
        logger.info("Computing statistics...")
        defect_stats = DefectStatistics()
        statistics = defect_stats.compute_batch_statistics(mask_paths)
        
        # Save statistics JSON
        stats_path = output_dir / 'statistics.json'
        with open(stats_path, 'w') as f:
            json.dump(statistics, f, indent=2)
        logger.info(f"Statistics saved to: {stats_path}")
        
        # 2. Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = DefectVisualizer()
        
        figures = {}
        chart_paths = {}
        
        # Defect size distribution
        all_defect_areas = []
        for img_stats in statistics.get('per_image_stats', []):
            all_defect_areas.extend(img_stats.get('defect_areas', []))
        
        if all_defect_areas:
            fig1 = visualizer.plot_defect_size_distribution(
                all_defect_areas,
                output_path=str(output_dir / 'defect_size_distribution.png')
            )
            figures['size_distribution'] = fig1
            chart_paths['size_distribution'] = str(output_dir / 'defect_size_distribution.png')
        
        # Defect count per image
        defect_counts = [stats['num_defects'] for stats in statistics.get('per_image_stats', [])]
        if defect_counts:
            fig2 = visualizer.plot_defect_count_per_image(
                defect_counts,
                output_path=str(output_dir / 'defect_count_per_image.png')
            )
            figures['defect_counts'] = fig2
            chart_paths['defect_counts'] = str(output_dir / 'defect_count_per_image.png')
        
        # Coverage ratio distribution
        coverage_ratios = [stats['coverage_ratio'] for stats in statistics.get('per_image_stats', [])
                          if stats['coverage_ratio'] > 0]
        if coverage_ratios:
            fig3 = visualizer.plot_coverage_ratio_distribution(
                coverage_ratios,
                output_path=str(output_dir / 'coverage_ratio_distribution.png')
            )
            figures['coverage_ratio'] = fig3
            chart_paths['coverage_ratio'] = str(output_dir / 'coverage_ratio_distribution.png')
        
        # 3. Generate reports
        report_paths = {}
        
        if 'excel' in report_formats:
            try:
                if self.excel_generator is None:
                    self.excel_generator = ExcelReportGenerator()
                
                excel_path = output_dir / 'defect_report.xlsx'
                self.excel_generator.generate_defect_report(statistics, str(excel_path))
                report_paths['excel'] = str(excel_path)
            except Exception as e:
                logger.error(f"Failed to generate Excel report: {e}")
        
        if 'pdf' in report_formats:
            try:
                if self.pdf_generator is None:
                    self.pdf_generator = PDFReportGenerator()
                
                pdf_path = output_dir / 'defect_report.pdf'
                self.pdf_generator.generate_defect_report(
                    statistics,
                    list(figures.values()),
                    str(pdf_path)
                )
                report_paths['pdf'] = str(pdf_path)
            except Exception as e:
                logger.error(f"Failed to generate PDF report: {e}")
        
        if 'html' in report_formats:
            try:
                html_path = output_dir / 'defect_report.html'
                self.html_generator.generate_defect_report(
                    statistics,
                    chart_paths,
                    str(html_path)
                )
                report_paths['html'] = str(html_path)
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")
        
        logger.info("Report generation complete!")
        logger.info(f"Reports saved to: {output_dir}")
        
        return {
            'statistics': statistics,
            'figures': figures,
            'report_paths': report_paths
        }
