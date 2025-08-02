#!/usr/bin/env python3
"""
Generate interactive metrics dashboard for photon-mlir-bridge project.

This script creates an HTML dashboard displaying project metrics,
trends, and insights from the collected data.
"""

import json
import os
import sys
import argparse
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
from jinja2 import Template
import base64
from io import BytesIO


class MetricsDashboard:
    """Generate interactive metrics dashboard."""
    
    def __init__(self, metrics_file: str = '.github/project-metrics.json'):
        self.logger = self._setup_logging()
        self.metrics_file = Path(metrics_file)
        self.metrics_data = self._load_metrics()
        self.output_dir = Path('dashboard')
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Load metrics configuration and data."""
        try:
            with open(self.metrics_file) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Metrics file not found: {self.metrics_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in metrics file: {e}")
            sys.exit(1)
    
    def create_output_directory(self):
        """Create output directory for dashboard files."""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / 'assets').mkdir(exist_ok=True)
        (self.output_dir / 'charts').mkdir(exist_ok=True)
    
    def generate_metric_chart(self, category: str, metric_name: str, metric_data: Dict[str, Any]) -> str:
        """Generate chart for a specific metric."""
        try:
            current = metric_data.get('current', 0)
            target = metric_data.get('target', 0)
            warning = metric_data.get('warning_threshold', 0)
            critical = metric_data.get('critical_threshold', 0)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Generate some sample historical data for demonstration
            # In a real implementation, this would come from a time series database
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            
            # Simulate trend data based on current trend
            trend = metric_data.get('trend', 'stable')
            if trend == 'improving':
                base_values = [current * (0.9 + 0.2 * (i / 30)) for i in range(30)]
            elif trend == 'degrading':
                base_values = [current * (1.1 - 0.2 * (i / 30)) for i in range(30)]
            else:  # stable
                base_values = [current + (current * 0.1 * (0.5 - abs(0.5 - i/30))) for i in range(30)]
            
            # Add some noise
            import numpy as np
            np.random.seed(42)  # For reproducible results
            values = [max(0, val + np.random.normal(0, abs(val) * 0.05)) for val in base_values]
            
            # Plot the metric over time
            ax.plot(dates, values, linewidth=2, label='Actual', color='#2E86AB')
            ax.axhline(y=target, color='green', linestyle='--', label='Target', alpha=0.7)
            
            # Add threshold lines based on metric type
            if warning and critical:
                if metric_name in ['test_coverage', 'github_stars', 'build_success_rate']:
                    # Higher is better
                    if warning < target:
                        ax.axhline(y=warning, color='orange', linestyle=':', label='Warning', alpha=0.7)
                    if critical < warning:
                        ax.axhline(y=critical, color='red', linestyle=':', label='Critical', alpha=0.7)
                else:
                    # Lower is better
                    if warning > target:
                        ax.axhline(y=warning, color='orange', linestyle=':', label='Warning', alpha=0.7)
                    if critical > warning:
                        ax.axhline(y=critical, color='red', linestyle=':', label='Critical', alpha=0.7)
            
            # Formatting
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel(metric_data.get('unit', 'Value'))
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            plt.xticks(rotation=45)
            
            # Tight layout
            plt.tight_layout()
            
            # Save to base64 string for embedding in HTML
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Error generating chart for {metric_name}: {e}")
            return ""
    
    def generate_category_summary_chart(self, category: str, category_data: Dict[str, Any]) -> str:
        """Generate summary chart for a metric category."""
        try:
            measurements = category_data.get('measurements', {})
            
            if not measurements:
                return ""
            
            # Create a radar chart for the category
            metrics = []
            current_values = []
            target_values = []
            normalized_current = []
            normalized_target = []
            
            for metric_name, metric_data in list(measurements.items())[:8]:  # Limit to 8 metrics
                current = metric_data.get('current', 0)
                target = metric_data.get('target', 0)
                
                if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    metrics.append(metric_name.replace('_', ' ').title())
                    current_values.append(current)
                    target_values.append(target)
                    
                    # Normalize values for radar chart
                    if target > 0:
                        if metric_name in ['complexity', 'error_rate', 'critical_vulnerabilities']:
                            # Lower is better - invert the scale
                            norm_current = max(0, 100 - (current / target * 100))
                            norm_target = 100
                        else:
                            # Higher is better
                            norm_current = min(100, current / target * 100)
                            norm_target = 100
                    else:
                        norm_current = 50  # Default middle value
                        norm_target = 100
                    
                    normalized_current.append(norm_current)
                    normalized_target.append(norm_target)
            
            if not metrics:
                return ""
            
            # Create radar chart
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            # Calculate angles for each metric
            angles = [n / len(metrics) * 2 * 3.14159 for n in range(len(metrics))]
            angles += angles[:1]  # Complete the circle
            
            # Add values
            normalized_current += normalized_current[:1]
            normalized_target += normalized_target[:1]
            
            # Plot
            ax.plot(angles, normalized_current, 'o-', linewidth=2, label='Current', color='#2E86AB')
            ax.fill(angles, normalized_current, alpha=0.25, color='#2E86AB')
            ax.plot(angles, normalized_target, 'o-', linewidth=2, label='Target', color='green', linestyle='--')
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 100)
            ax.set_yticks([20, 40, 60, 80, 100])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True)
            
            # Title and legend
            ax.set_title(f'{category.replace("_", " ").title()} Metrics Overview', 
                        pad=20, fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            
            plt.tight_layout()
            
            # Save to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            self.logger.error(f"Error generating category chart for {category}: {e}")
            return ""
    
    def generate_overall_health_score(self) -> Dict[str, Any]:
        """Calculate overall project health score."""
        try:
            total_score = 0
            category_scores = {}
            weights = {
                'code_quality': 0.25,
                'development_velocity': 0.20,
                'build_system': 0.15,
                'performance': 0.15,
                'security': 0.15,
                'community': 0.10
            }
            
            for category, category_data in self.metrics_data.get('metrics', {}).items():
                measurements = category_data.get('measurements', {})
                category_score = 0
                valid_metrics = 0
                
                for metric_name, metric_data in measurements.items():
                    current = metric_data.get('current', 0)
                    target = metric_data.get('target', 0)
                    
                    if isinstance(current, (int, float)) and isinstance(target, (int, float)) and target > 0:
                        # Calculate score based on how close current is to target
                        if metric_name in ['complexity', 'error_rate', 'critical_vulnerabilities', 'mean_time_to_recovery']:
                            # Lower is better
                            score = max(0, min(100, 100 - (current / target - 1) * 100))
                        else:
                            # Higher is better
                            score = max(0, min(100, (current / target) * 100))
                        
                        category_score += score
                        valid_metrics += 1
                
                if valid_metrics > 0:
                    category_scores[category] = category_score / valid_metrics
                    weight = weights.get(category, 0.1)
                    total_score += category_scores[category] * weight
            
            # Determine grade based on score
            if total_score >= 90:
                grade = 'A'
                status = 'excellent'
            elif total_score >= 80:
                grade = 'B'
                status = 'good'
            elif total_score >= 70:
                grade = 'C'
                status = 'fair'
            elif total_score >= 60:
                grade = 'D'
                status = 'poor'
            else:
                grade = 'F'
                status = 'critical'
            
            return {
                'overall_score': round(total_score, 1),
                'grade': grade,
                'status': status,
                'category_scores': {k: round(v, 1) for k, v in category_scores.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating health score: {e}")
            return {
                'overall_score': 0,
                'grade': 'F',
                'status': 'unknown',
                'category_scores': {}
            }
    
    def generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML."""
        # Calculate health score
        health_score = self.generate_overall_health_score()
        
        # Generate charts for each category
        category_charts = {}
        detailed_charts = {}
        
        for category, category_data in self.metrics_data.get('metrics', {}).items():
            # Category overview chart
            category_charts[category] = self.generate_category_summary_chart(category, category_data)
            
            # Individual metric charts
            detailed_charts[category] = {}
            measurements = category_data.get('measurements', {})
            
            for metric_name, metric_data in list(measurements.items())[:5]:  # Limit to 5 charts per category
                chart_data = self.generate_metric_chart(category, metric_name, metric_data)
                if chart_data:
                    detailed_charts[category][metric_name] = chart_data
        
        # HTML template
        html_template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ project_name }} - Metrics Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }
        
        .health-score {
            background: white;
            margin: 2rem 0;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .score-circle {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            margin: 0 auto 1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            font-weight: bold;
            color: white;
        }
        
        .score-excellent { background: linear-gradient(135deg, #4CAF50, #45a049); }
        .score-good { background: linear-gradient(135deg, #2196F3, #1976D2); }
        .score-fair { background: linear-gradient(135deg, #FF9800, #F57C00); }
        .score-poor { background: linear-gradient(135deg, #FF5722, #D84315); }
        .score-critical { background: linear-gradient(135deg, #F44336, #C62828); }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
            margin: 2rem 0;
        }
        
        .metric-card {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
        }
        
        .metric-card-header {
            background: #f8f9fa;
            padding: 1rem;
            border-bottom: 1px solid #e9ecef;
        }
        
        .metric-card-header h3 {
            color: #495057;
            font-size: 1.2rem;
        }
        
        .metric-card-body {
            padding: 1rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        
        .metric-trend {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.5rem;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .trend-improving { background: #d4edda; color: #155724; }
        .trend-stable { background: #d1ecf1; color: #0c5460; }
        .trend-degrading { background: #f8d7da; color: #721c24; }
        
        .chart-container {
            margin: 1rem 0;
            text-align: center;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 5px;
        }
        
        .category-section {
            margin: 3rem 0;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .category-header {
            background: linear-gradient(135deg, #6c5ce7, #a29bfe);
            color: white;
            padding: 1.5rem;
        }
        
        .category-header h2 {
            font-size: 1.5rem;
        }
        
        .category-body {
            padding: 2rem;
        }
        
        .tabs {
            display: flex;
            border-bottom: 2px solid #e9ecef;
            margin-bottom: 2rem;
        }
        
        .tab {
            padding: 1rem 2rem;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            color: #6c757d;
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        
        .tab.active {
            color: #495057;
            border-bottom-color: #6c5ce7;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .footer {
            background: #343a40;
            color: white;
            text-align: center;
            padding: 2rem;
            margin-top: 3rem;
        }
        
        .timestamp {
            color: #6c757d;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2rem; }
            .metrics-grid { grid-template-columns: 1fr; }
            .tabs { flex-direction: column; }
            .tab { text-align: left; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>{{ project_name }}</h1>
            <p>Project Metrics Dashboard</p>
        </div>
    </div>
    
    <div class="container">
        <!-- Health Score Section -->
        <div class="health-score">
            <div class="score-circle score-{{ health_score.status }}">
                {{ health_score.grade }}
            </div>
            <h2>Overall Health Score: {{ health_score.overall_score }}%</h2>
            <p>Project status: <strong>{{ health_score.status.title() }}</strong></p>
        </div>
        
        <!-- Category Scores -->
        <div class="metrics-grid">
            {% for category, score in health_score.category_scores.items() %}
            <div class="metric-card">
                <div class="metric-card-header">
                    <h3>{{ category.replace('_', ' ').title() }}</h3>
                </div>
                <div class="metric-card-body">
                    <div class="metric-value">{{ score }}%</div>
                </div>
            </div>
            {% endfor %}
        </div>
        
        <!-- Detailed Category Sections -->
        {% for category, category_data in metrics.items() %}
        <div class="category-section">
            <div class="category-header">
                <h2>{{ category.replace('_', ' ').title() }} Metrics</h2>
                <p>{{ category_data.description }}</p>
            </div>
            <div class="category-body">
                <div class="tabs">
                    <button class="tab active" onclick="showTab('{{ category }}-overview')">Overview</button>
                    <button class="tab" onclick="showTab('{{ category }}-details')">Detailed Charts</button>
                </div>
                
                <div id="{{ category }}-overview" class="tab-content active">
                    {% if category_charts[category] %}
                    <div class="chart-container">
                        <img src="{{ category_charts[category] }}" alt="{{ category }} overview chart">
                    </div>
                    {% endif %}
                    
                    <div class="metrics-grid">
                        {% for metric_name, metric_data in category_data.measurements.items() %}
                        <div class="metric-card">
                            <div class="metric-card-header">
                                <h3>{{ metric_name.replace('_', ' ').title() }}</h3>
                            </div>
                            <div class="metric-card-body">
                                <div class="metric-value">
                                    {{ metric_data.current }}
                                    {% if metric_data.unit %}{{ metric_data.unit }}{% endif %}
                                </div>
                                <div>Target: {{ metric_data.target }}</div>
                                <div>
                                    <span class="metric-trend trend-{{ metric_data.trend }}">
                                        {{ metric_data.trend.title() }}
                                    </span>
                                </div>
                                <div class="timestamp">
                                    Last updated: {{ metric_data.last_measured or 'Never' }}
                                </div>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div id="{{ category }}-details" class="tab-content">
                    {% for metric_name, chart_data in detailed_charts[category].items() %}
                    <div class="chart-container">
                        <h4>{{ metric_name.replace('_', ' ').title() }}</h4>
                        <img src="{{ chart_data }}" alt="{{ metric_name }} trend chart">
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
        {% endfor %}
    </div>
    
    <div class="footer">
        <div class="container">
            <p>&copy; {{ current_year }} {{ project_name }} Project</p>
            <p class="timestamp">Dashboard generated: {{ generation_time }}</p>
            <p>🤖 <em>Automated metrics dashboard</em></p>
        </div>
    </div>
    
    <script>
        function showTab(tabId) {
            // Hide all tab contents in the same category
            const category = tabId.split('-')[0];
            document.querySelectorAll(`[id^="${category}-"]`).forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Remove active class from all tabs in the same category
            document.querySelectorAll('.tab').forEach(tab => {
                if (tab.onclick.toString().includes(category)) {
                    tab.classList.remove('active');
                }
            });
            
            // Show selected tab
            document.getElementById(tabId).classList.add('active');
            
            // Add active class to clicked tab
            event.target.classList.add('active');
        }
        
        // Auto-refresh dashboard every 5 minutes
        setTimeout(() => {
            location.reload();
        }, 300000);
    </script>
</body>
</html>
        """)
        
        return html_template.render(
            project_name=self.metrics_data.get('project', {}).get('name', 'Project'),
            health_score=health_score,
            metrics=self.metrics_data.get('metrics', {}),
            category_charts=category_charts,
            detailed_charts=detailed_charts,
            generation_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
            current_year=datetime.now().year
        )
    
    def generate_dashboard(self, output_file: str = 'dashboard/index.html') -> bool:
        """Generate the complete metrics dashboard."""
        self.logger.info("Generating metrics dashboard...")
        
        try:
            # Create output directory
            self.create_output_directory()
            
            # Generate HTML
            html_content = self.generate_dashboard_html()
            
            # Write to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Dashboard generated successfully: {output_path.absolute()}")
            
            # Also generate a JSON summary for API consumption
            summary = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'health_score': self.generate_overall_health_score(),
                'project_info': self.metrics_data.get('project', {}),
                'metrics_summary': {}
            }
            
            for category, category_data in self.metrics_data.get('metrics', {}).items():
                measurements = category_data.get('measurements', {})
                summary['metrics_summary'][category] = {
                    'count': len(measurements),
                    'metrics': {name: {
                        'current': data.get('current', 0),
                        'target': data.get('target', 0),
                        'trend': data.get('trend', 'unknown')
                    } for name, data in measurements.items()}
                }
            
            with open(output_path.parent / 'dashboard-data.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard: {e}")
            return False


def main():
    """Main entry point for dashboard generation."""
    parser = argparse.ArgumentParser(description='Generate metrics dashboard')
    parser.add_argument('--metrics-file', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--output', default='dashboard/index.html',
                       help='Output file for the dashboard')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        dashboard = MetricsDashboard(args.metrics_file)
        success = dashboard.generate_dashboard(args.output)
        
        if success:
            print(f"Dashboard generated successfully: {Path(args.output).absolute()}")
            print(f"Open in browser: file://{Path(args.output).absolute()}")
            sys.exit(0)
        else:
            print("Dashboard generation failed")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()