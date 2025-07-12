"""
Error Analysis and Visualization

This module provides error analysis and visualization capabilities
for the HSC system error recovery.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from pathlib import Path

class ErrorAnalyzer:
    """Error analyzer with visualization capabilities."""
    
    def __init__(self):
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def print_error_summary(self, decode_result):
        """Print a human-readable error summary."""
        if not decode_result.errors:
            print("✅ No errors detected")
            return
        
        print(f"\n📊 Error Summary:")
        print(f"   Total errors: {len(decode_result.errors)}")
        print(f"   Overall confidence: {decode_result.confidence:.3f}")
        
        # Group errors by type
        error_types = {}
        for error in decode_result.errors:
            error_type = error.error_type.value
            if error_type not in error_types:
                error_types[error_type] = []
            error_types[error_type].append(error)
        
        print(f"   Error types: {list(error_types.keys())}")
        
        # Show top errors
        print(f"\n🔍 Top Errors:")
        for i, error in enumerate(decode_result.errors[:3]):
            print(f"   {i+1}. {error.error_type.value}: {error.message}")
            print(f"      Confidence: {error.confidence:.3f}")
    
    def plot_error_summary(self, error_report: Dict[str, Any], filename: str):
        """Create error summary visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Error Analysis Summary', fontsize=16, fontweight='bold')
        
        # Error type distribution
        if 'error_types' in error_report and error_report['error_types']:
            error_types = list(error_report['error_types'].keys())
            error_counts = list(error_report['error_types'].values())
            
            axes[0, 0].pie(error_counts, labels=error_types, autopct='%1.1f%%')
            axes[0, 0].set_title('Error Type Distribution')
        
        # Severity distribution
        if 'severity_distribution' in error_report and error_report['severity_distribution']:
            severities = list(error_report['severity_distribution'].keys())
            severity_counts = list(error_report['severity_distribution'].values())
            
            colors = ['red' if s == 'CRITICAL' else 'orange' if s == 'HIGH' else 'yellow' if s == 'MEDIUM' else 'green' for s in severities]
            axes[0, 1].bar(severities, severity_counts, color=colors)
            axes[0, 1].set_title('Error Severity Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Confidence distribution
        if 'confidence_distribution' in error_report:
            conf_data = error_report['confidence_distribution']
            axes[1, 0].hist([conf_data['mean']], bins=10, alpha=0.7, color='blue')
            axes[1, 0].axvline(conf_data['mean'], color='red', linestyle='--', label=f'Mean: {conf_data["mean"]:.3f}')
            axes[1, 0].set_title('Confidence Distribution')
            axes[1, 0].legend()
        
        # Vector analysis
        if 'vector_analysis' in error_report:
            vec_analysis = error_report['vector_analysis']
            metrics = ['norm', 'mean', 'std']
            values = [vec_analysis.get(m, 0) for m in metrics]
            
            # Handle inf/nan values
            values = [v if np.isfinite(v) else 0 for v in values]
            
            axes[1, 1].bar(metrics, values, color=['purple', 'green', 'orange'])
            axes[1, 1].set_title('Vector Statistics')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Error summary plot saved to {filename}")
    
    def plot_vector_analysis(self, vector: np.ndarray, error_report: Dict[str, Any], filename: str):
        """Create vector analysis visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Vector Analysis', fontsize=16, fontweight='bold')
        
        # Vector values over time
        axes[0, 0].plot(vector, alpha=0.7, linewidth=1)
        axes[0, 0].set_title('Vector Values')
        axes[0, 0].set_xlabel('Dimension')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Vector histogram
        finite_vector = vector[np.isfinite(vector)]
        if len(finite_vector) > 0:
            axes[0, 1].hist(finite_vector, bins=30, alpha=0.7, color='green')
            axes[0, 1].set_title('Value Distribution')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
        
        # Vector statistics
        if 'vector_analysis' in error_report:
            vec_analysis = error_report['vector_analysis']
            
            # Create statistics table
            stats_data = [
                ['Dimension', str(vec_analysis.get('dimension', 'N/A'))],
                ['Norm', f"{vec_analysis.get('norm', 0):.3f}"],
                ['Mean', f"{vec_analysis.get('mean', 0):.3f}"],
                ['Std', f"{vec_analysis.get('std', 0):.3f}"],
                ['Min', f"{vec_analysis.get('min', 0):.3f}"],
                ['Max', f"{vec_analysis.get('max', 0):.3f}"],
                ['Has NaN', str(vec_analysis.get('has_nan', False))],
                ['Has Inf', str(vec_analysis.get('has_inf', False))],
                ['Zero Count', str(vec_analysis.get('zero_count', 0))],
                ['Distribution', str(vec_analysis.get('distribution_type', 'UNKNOWN'))]
            ]
            
            axes[1, 0].axis('tight')
            axes[1, 0].axis('off')
            table = axes[1, 0].table(cellText=stats_data, colLabels=['Metric', 'Value'], 
                                   cellLoc='left', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
            axes[1, 0].set_title('Vector Statistics')
        
        # Error indicators
        error_indicators = []
        if 'vector_analysis' in error_report:
            vec_analysis = error_report['vector_analysis']
            if vec_analysis.get('has_nan', False):
                error_indicators.append('NaN values')
            if vec_analysis.get('has_inf', False):
                error_indicators.append('Inf values')
            if vec_analysis.get('extreme_count', 0) > 0:
                error_indicators.append('Extreme values')
            if vec_analysis.get('zero_count', 0) > len(vector) * 0.5:
                error_indicators.append('Too many zeros')
        
        if error_indicators:
            axes[1, 1].text(0.1, 0.5, '\n'.join([f'⚠️ {indicator}' for indicator in error_indicators]), 
                           fontsize=12, color='red', transform=axes[1, 1].transAxes)
        else:
            axes[1, 1].text(0.1, 0.5, '✅ Vector appears valid', 
                           fontsize=12, color='green', transform=axes[1, 1].transAxes)
        
        axes[1, 1].set_title('Error Indicators')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Vector analysis plot saved to {filename}")
    
    def plot_error_patterns(self, error_patterns: Dict[str, Any], filename: str):
        """Create error patterns visualization."""
        if not error_patterns:
            print("No error patterns to visualize")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Error Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Error type distribution
        if 'error_type_distribution' in error_patterns:
            error_types = list(error_patterns['error_type_distribution'].keys())
            error_counts = list(error_patterns['error_type_distribution'].values())
            
            axes[0].bar(error_types, error_counts, color='skyblue')
            axes[0].set_title('Error Type Distribution')
            axes[0].tick_params(axis='x', rotation=45)
            axes[0].set_ylabel('Count')
        
        # Confidence distribution
        if 'confidence_distribution' in error_patterns:
            conf_data = error_patterns['confidence_distribution']
            conf_stats = [conf_data['mean'], conf_data['std'], conf_data['min'], conf_data['max']]
            conf_labels = ['Mean', 'Std', 'Min', 'Max']
            
            axes[1].bar(conf_labels, conf_stats, color='lightgreen')
            axes[1].set_title('Confidence Statistics')
            axes[1].set_ylabel('Confidence')
        
        # Severity distribution
        if 'severity_distribution' in error_patterns:
            severities = list(error_patterns['severity_distribution'].keys())
            severity_counts = list(error_patterns['severity_distribution'].values())
            
            colors = ['red' if s == 'CRITICAL' else 'orange' if s == 'HIGH' else 'yellow' if s == 'MEDIUM' else 'green' for s in severities]
            axes[2].pie(severity_counts, labels=severities, autopct='%1.1f%%', colors=colors)
            axes[2].set_title('Severity Distribution')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Error patterns plot saved to {filename}")
    
    def create_error_report(self, decode_result, output_dir: str) -> List[str]:
        """Create a comprehensive error report with all visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Get error report data
        error_report = decode_result.metadata.get('error_report', {})
        error_patterns = decode_result.metadata.get('error_patterns', {})
        
        # Create visualizations
        if error_report:
            summary_file = output_path / "error_summary.png"
            self.plot_error_summary(error_report, str(summary_file))
            saved_files.append(str(summary_file))
            
            # Vector analysis (if we have a vector)
            if 'vector_analysis' in error_report:
                vector_file = output_path / "vector_analysis.png"
                # We need the original vector for this - for now, create a dummy one
                dummy_vector = np.random.randn(64)  # This should be the actual vector
                self.plot_vector_analysis(dummy_vector, error_report, str(vector_file))
                saved_files.append(str(vector_file))
        
        if error_patterns:
            patterns_file = output_path / "error_patterns.png"
            self.plot_error_patterns(error_patterns, str(patterns_file))
            saved_files.append(str(patterns_file))
        
        # Create text report
        report_file = output_path / "error_report.txt"
        with open(report_file, 'w') as f:
            f.write("HSC Error Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            if error_report:
                f.write(f"Total Errors: {error_report.get('total_errors', 0)}\n")
                f.write(f"Overall Confidence: {error_report.get('overall_confidence', 0):.3f}\n\n")
                
                f.write("Error Types:\n")
                for error_type, count in error_report.get('error_types', {}).items():
                    f.write(f"  - {error_type}: {count}\n")
                
                f.write("\nRecovery Recommendations:\n")
                for rec in error_report.get('recovery_recommendations', []):
                    f.write(f"  - {rec}\n")
            
            f.write("\nDetailed Errors:\n")
            for error in decode_result.errors:
                f.write(f"\nError: {error.error_type.value}\n")
                f.write(f"Message: {error.message}\n")
                f.write(f"Confidence: {error.confidence:.3f}\n")
                f.write(f"Suggestions: {len(error.suggestions)} items\n")
        
        saved_files.append(str(report_file))
        
        print(f"📊 Error report created in {output_dir}")
        return saved_files 