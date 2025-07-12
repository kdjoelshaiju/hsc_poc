"""
Error Analysis Visualization

This module provides comprehensive visualization tools for analyzing
decoding errors, vector characteristics, and recovery patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional
import seaborn as sns
from dataclasses import dataclass

@dataclass
class ErrorVisualizationConfig:
    """Configuration for error visualization."""
    figure_size: tuple = (15, 10)
    color_palette: str = "viridis"
    save_format: str = "png"
    dpi: int = 300
    show_plots: bool = True

class ErrorAnalyzer:
    """Comprehensive error analysis and visualization tool."""
    
    def __init__(self, config: Optional[ErrorVisualizationConfig] = None):
        self.config = config or ErrorVisualizationConfig()
        self.setup_style()
    
    def setup_style(self):
        """Setup matplotlib style for error analysis plots."""
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['font.size'] = 10
        # Make plots non-blocking
        plt.rcParams['backend'] = 'Agg'  # Use non-interactive backend
    
    def plot_error_summary(self, error_report: Dict[str, Any], save_path: Optional[str] = None):
        """Create a comprehensive error summary visualization."""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Error Analysis Summary', fontsize=16, fontweight='bold')
            
            # 1. Error Type Distribution
            if 'error_types' in error_report and error_report['error_types']:
                error_types = list(error_report['error_types'].keys())
                error_counts = list(error_report['error_types'].values())
                
                axes[0, 0].pie(error_counts, labels=error_types, autopct='%1.1f%%', startangle=90)
                axes[0, 0].set_title('Error Type Distribution')
            
            # 2. Severity Distribution
            if 'severity_distribution' in error_report and error_report['severity_distribution']:
                severities = list(error_report['severity_distribution'].keys())
                severity_counts = list(error_report['severity_distribution'].values())
                
                colors = ['red' if s == 'CRITICAL' else 'orange' if s == 'HIGH' else 'yellow' if s == 'MEDIUM' else 'green' for s in severities]
                axes[0, 1].bar(severities, severity_counts, color=colors)
                axes[0, 1].set_title('Error Severity Distribution')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Confidence Distribution
            if 'confidence_distribution' in error_report:
                conf_data = error_report['confidence_distribution']
                axes[0, 2].hist([conf_data['mean']], bins=10, alpha=0.7, color='blue')
                axes[0, 2].axvline(conf_data['mean'], color='red', linestyle='--', label=f'Mean: {conf_data["mean"]:.3f}')
                axes[0, 2].set_title('Confidence Distribution')
                axes[0, 2].legend()
            
            # 4. Vector Analysis
            if 'vector_analysis' in error_report:
                vec_analysis = error_report['vector_analysis']
                vec_stats = ['norm', 'mean', 'std', 'min', 'max']
                vec_values = [vec_analysis.get(stat, 0) for stat in vec_stats]
                
                axes[1, 0].bar(vec_stats, vec_values, color='skyblue')
                axes[1, 0].set_title('Vector Statistics')
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 5. Error Timeline
            if 'detailed_errors' in error_report:
                detailed_errors = error_report['detailed_errors']
                if detailed_errors:
                    indices = [e['index'] for e in detailed_errors]
                    confidences = [e['confidence'] for e in detailed_errors]
                    severities = [e['severity'] for e in detailed_errors]
                    
                    colors = ['red' if s == 'CRITICAL' else 'orange' if s == 'HIGH' else 'yellow' if s == 'MEDIUM' else 'green' for s in severities]
                    axes[1, 1].scatter(indices, confidences, c=colors, s=100, alpha=0.7)
                    axes[1, 1].set_xlabel('Error Index')
                    axes[1, 1].set_ylabel('Confidence')
                    axes[1, 1].set_title('Error Confidence Timeline')
            
            # 6. Recovery Recommendations
            if 'recovery_recommendations' in error_report:
                recommendations = error_report['recovery_recommendations']
                if recommendations:
                    axes[1, 2].text(0.1, 0.9, 'Recovery Recommendations:', fontsize=12, fontweight='bold', transform=axes[1, 2].transAxes)
                    for i, rec in enumerate(recommendations[:5]):  # Show first 5 recommendations
                        axes[1, 2].text(0.1, 0.8 - i*0.15, f'• {rec}', fontsize=9, transform=axes[1, 2].transAxes, wrap=True)
                    axes[1, 2].set_xlim(0, 1)
                    axes[1, 2].set_ylim(0, 1)
                    axes[1, 2].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                print(f"Error summary plot saved to {save_path}")
            
            plt.close()  # Always close the figure to prevent memory leaks
            
        except Exception as e:
            print(f"Error creating error summary plot: {e}")
            plt.close('all')  # Close all figures on error
    
    def plot_vector_analysis(self, vector: np.ndarray, error_report: Dict[str, Any], 
                           save_path: Optional[str] = None):
        """Create detailed vector analysis visualization."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Vector Analysis', fontsize=16, fontweight='bold')
            
            # Check if vector has valid data
            valid_vector = np.isfinite(vector)
            if not np.any(valid_vector):
                # Handle all-NaN or all-inf vectors
                axes[0, 0].text(0.5, 0.5, 'Vector contains no valid data\n(all NaN or inf)', 
                               ha='center', va='center', transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Vector Value Distribution')
                
                axes[0, 1].text(0.5, 0.5, 'No valid components to plot', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Vector Components')
                
                axes[1, 0].text(0.5, 0.5, 'No valid statistics', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Vector Statistics')
                
                axes[1, 1].text(0.5, 0.5, 'No error correlation data', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Error vs Vector Characteristics')
                
            else:
                # Use only valid data for plotting
                clean_vector = vector[valid_vector]
                
                # 1. Vector Distribution
                if len(clean_vector) > 0:
                    axes[0, 0].hist(clean_vector, bins=min(50, len(clean_vector)//2), alpha=0.7, color='blue', edgecolor='black')
                    axes[0, 0].axvline(np.mean(clean_vector), color='red', linestyle='--', label=f'Mean: {np.mean(clean_vector):.3f}')
                    axes[0, 0].axvline(np.median(clean_vector), color='green', linestyle='--', label=f'Median: {np.median(clean_vector):.3f}')
                    axes[0, 0].set_xlabel('Vector Values')
                    axes[0, 0].set_ylabel('Frequency')
                    axes[0, 0].set_title('Vector Value Distribution (Valid Data Only)')
                    axes[0, 0].legend()
                else:
                    axes[0, 0].text(0.5, 0.5, 'No valid data for histogram', 
                                   ha='center', va='center', transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Vector Value Distribution')
                
                # 2. Vector Components
                component_indices = np.arange(len(vector))
                valid_indices = component_indices[valid_vector]
                
                if len(valid_indices) > 0:
                    axes[0, 1].plot(valid_indices, clean_vector, 'b-', alpha=0.7, linewidth=1)
                    scatter = axes[0, 1].scatter(valid_indices, clean_vector, c=clean_vector, cmap='viridis', s=20, alpha=0.8)
                    axes[0, 1].set_xlabel('Component Index')
                    axes[0, 1].set_ylabel('Value')
                    axes[0, 1].set_title('Vector Components (Valid Data Only)')
                    axes[0, 1].grid(True, alpha=0.3)
                    plt.colorbar(scatter, ax=axes[0, 1])
                else:
                    axes[0, 1].text(0.5, 0.5, 'No valid components to plot', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
                    axes[0, 1].set_title('Vector Components')
                
                # 3. Vector Statistics
                if 'vector_analysis' in error_report:
                    vec_analysis = error_report['vector_analysis']
                    stats = ['Dimension', 'Norm', 'Mean', 'Std', 'Min', 'Max']
                    values = [
                        vec_analysis.get('dimension', 0),
                        vec_analysis.get('norm', 0),
                        vec_analysis.get('mean', 0),
                        vec_analysis.get('std', 0),
                        vec_analysis.get('min', 0),
                        vec_analysis.get('max', 0)
                    ]
                    
                    # Filter out invalid values for display
                    valid_stats = []
                    valid_values = []
                    for stat, value in zip(stats, values):
                        if np.isfinite(value):
                            valid_stats.append(stat)
                            valid_values.append(value)
                    
                    if valid_stats:
                        bars = axes[1, 0].bar(valid_stats, valid_values, color='lightcoral')
                        axes[1, 0].set_title('Vector Statistics (Valid Values)')
                        axes[1, 0].tick_params(axis='x', rotation=45)
                        
                        # Add value labels on bars
                        for bar, value in zip(bars, valid_values):
                            height = bar.get_height()
                            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                                           f'{value:.3f}', ha='center', va='bottom')
                    else:
                        axes[1, 0].text(0.5, 0.5, 'No valid statistics', 
                                       ha='center', va='center', transform=axes[1, 0].transAxes)
                        axes[1, 0].set_title('Vector Statistics')
                
                # 4. Error Correlation
                if 'detailed_errors' in error_report:
                    detailed_errors = error_report['detailed_errors']
                    if detailed_errors and 'vector_slice_info' in detailed_errors[0]:
                        error_indices = []
                        error_confidences = []
                        vector_norms = []
                        
                        for error in detailed_errors:
                            if error['vector_slice_info']:
                                error_indices.append(error['index'])
                                error_confidences.append(error['confidence'])
                                vector_norms.append(error['vector_slice_info']['norm'])
                        
                        if error_indices and all(np.isfinite(v) for v in vector_norms):
                            scatter = axes[1, 1].scatter(error_confidences, vector_norms, 
                                                       c=error_indices, cmap='viridis', s=100, alpha=0.7)
                            axes[1, 1].set_xlabel('Error Confidence')
                            axes[1, 1].set_ylabel('Vector Slice Norm')
                            axes[1, 1].set_title('Error vs Vector Characteristics')
                            plt.colorbar(scatter, ax=axes[1, 1], label='Error Index')
                        else:
                            axes[1, 1].text(0.5, 0.5, 'No valid error correlation data', 
                                           ha='center', va='center', transform=axes[1, 1].transAxes)
                            axes[1, 1].set_title('Error vs Vector Characteristics')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                print(f"Vector analysis plot saved to {save_path}")
            
            plt.close()  # Always close the figure
            
        except Exception as e:
            print(f"Error creating vector analysis plot: {e}")
            plt.close('all')
    
    def plot_error_patterns(self, error_patterns: Dict[str, Any], save_path: Optional[str] = None):
        """Create error pattern visualization."""
        if not error_patterns:
            print("No error patterns to visualize")
            return
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle('Error Pattern Analysis', fontsize=16, fontweight='bold')
            
            # 1. Error Type Distribution
            if 'error_type_distribution' in error_patterns:
                error_types = list(error_patterns['error_type_distribution'].keys())
                type_counts = list(error_patterns['error_type_distribution'].values())
                
                axes[0].bar(error_types, type_counts, color='lightblue', edgecolor='black')
                axes[0].set_title('Error Type Distribution')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Add count labels
                for i, count in enumerate(type_counts):
                    axes[0].text(i, count + 0.1, str(count), ha='center', va='bottom')
            
            # 2. Confidence Distribution
            if 'confidence_distribution' in error_patterns:
                conf_data = error_patterns['confidence_distribution']
                conf_stats = ['Mean', 'Std', 'Min', 'Max']
                conf_values = [conf_data['mean'], conf_data['std'], conf_data['min'], conf_data['max']]
                
                bars = axes[1].bar(conf_stats, conf_values, color='lightgreen', edgecolor='black')
                axes[1].set_title('Confidence Statistics')
                
                # Add value labels
                for bar, value in zip(bars, conf_values):
                    height = bar.get_height()
                    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.3f}', ha='center', va='bottom')
            
            # 3. Severity Distribution
            if 'severity_distribution' in error_patterns:
                severities = list(error_patterns['severity_distribution'].keys())
                severity_counts = list(error_patterns['severity_distribution'].values())
                
                colors = ['red' if s == 'CRITICAL' else 'orange' if s == 'HIGH' else 'yellow' if s == 'MEDIUM' else 'green' for s in severities]
                axes[2].pie(severity_counts, labels=severities, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[2].set_title('Error Severity Distribution')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                print(f"Error patterns plot saved to {save_path}")
            
            plt.close()  # Always close the figure
            
        except Exception as e:
            print(f"Error creating error patterns plot: {e}")
            plt.close('all')
    
    def create_error_report(self, decode_result, save_dir: Optional[str] = None) -> Dict[str, str]:
        """Create a comprehensive error report with all visualizations."""
        if not decode_result.errors:
            print("No errors to analyze")
            return {}
        
        error_report = decode_result.metadata.get('error_report', {})
        error_patterns = decode_result.metadata.get('error_patterns', {})
        
        saved_files = {}
        
        # Create visualizations
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            
            # Error summary
            summary_path = os.path.join(save_dir, 'error_summary.png')
            self.plot_error_summary(error_report, summary_path)
            saved_files['error_summary'] = summary_path
            
            # Vector analysis
            vector_path = os.path.join(save_dir, 'vector_analysis.png')
            self.plot_vector_analysis(decode_result.metadata.get('vector_norm', np.array([])), 
                                   error_report, vector_path)
            saved_files['vector_analysis'] = vector_path
            
            # Error patterns
            patterns_path = os.path.join(save_dir, 'error_patterns.png')
            self.plot_error_patterns(error_patterns, patterns_path)
            saved_files['error_patterns'] = patterns_path
        
        return saved_files
    
    def print_error_summary(self, decode_result):
        """Print a detailed error summary to console."""
        if not decode_result.errors:
            print("✅ No errors detected")
            return
        
        error_report = decode_result.metadata.get('error_report', {})
        
        print("\n" + "="*60)
        print("🔍 ERROR ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total Errors: {error_report.get('total_errors', 0)}")
        print(f"Overall Confidence: {error_report.get('overall_confidence', 0):.3f}")
        print(f"Success: {'❌' if not decode_result.success else '✅'}")
        
        print("\n📊 Error Type Distribution:")
        for error_type, count in error_report.get('error_types', {}).items():
            print(f"  • {error_type}: {count}")
        
        print("\n⚠️ Severity Distribution:")
        for severity, count in error_report.get('severity_distribution', {}).items():
            print(f"  • {severity}: {count}")
        
        print("\n💡 Recovery Recommendations:")
        for i, rec in enumerate(error_report.get('recovery_recommendations', []), 1):
            print(f"  {i}. {rec}")
        
        print("\n🔧 Detailed Error Analysis:")
        for error in error_report.get('detailed_errors', []):
            print(f"  Error {error['index']}: {error['type']} ({error['severity']})")
            print(f"    Message: {error['message']}")
            print(f"    Confidence: {error['confidence']:.3f}")
            print(f"    Suggestions: {len(error['suggestions'])} items")
        
        print("="*60) 