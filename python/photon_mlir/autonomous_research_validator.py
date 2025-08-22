"""
Autonomous Research Validator - Generation 2 Enhancement
Advanced research validation with statistical analysis and peer-review preparation

This module implements comprehensive research validation capabilities including:
- Automated experimental design and execution
- Statistical significance testing with multiple correction methods
- Reproducibility validation across different environments
- Peer-review readiness assessment
- Benchmarking against state-of-the-art baselines
- Publication-quality result generation
"""

import asyncio
import numpy as np
import scipy.stats
import time
import logging
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile

from .core import TargetConfig, Device, Precision, PhotonicTensor
from .logging_config import get_global_logger
from .robust_error_handling import robust_execution


class ExperimentType(Enum):
    """Types of research experiments."""
    COMPARATIVE_STUDY = "comparative_study"
    ABLATION_STUDY = "ablation_study"
    SCALABILITY_ANALYSIS = "scalability_analysis"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    REPRODUCIBILITY_TEST = "reproducibility_test"
    NOVEL_ALGORITHM_VALIDATION = "novel_algorithm_validation"


class ValidationLevel(Enum):
    """Levels of validation rigor."""
    BASIC = "basic"
    STANDARD = "standard"
    RIGOROUS = "rigorous"
    PUBLICATION_READY = "publication_ready"


class StatisticalTest(Enum):
    """Statistical tests for validation."""
    T_TEST = "t_test"
    WELCH_T_TEST = "welch_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON = "wilcoxon"
    ANOVA = "anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    BOOTSTRAP = "bootstrap"


@dataclass
class ExperimentalCondition:
    """Defines experimental conditions."""
    name: str
    parameters: Dict[str, Any]
    description: str
    expected_outcome: Optional[str] = None


@dataclass
class ExperimentalResult:
    """Results from a single experimental run."""
    condition: ExperimentalCondition
    measurements: List[float]
    metadata: Dict[str, Any]
    timestamp: float
    environment_hash: str
    success: bool


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results."""
    test_type: StatisticalTest
    test_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    power: float
    sample_size: int
    significant: bool
    corrected_p_value: Optional[float] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    experiment_type: ExperimentType
    validation_level: ValidationLevel
    conditions: List[ExperimentalCondition]
    results: List[ExperimentalResult]
    statistical_analyses: List[StatisticalAnalysis]
    reproducibility_score: float
    peer_review_readiness: float
    recommendations: List[str]
    publication_artifacts: Dict[str, Any]


class AutonomousResearchValidator:
    """
    Autonomous research validation system with comprehensive statistical analysis.
    
    Provides rigorous validation of quantum-photonic research with automated
    experimental design, execution, and analysis suitable for peer review.
    """
    
    def __init__(self, 
                 validation_level: ValidationLevel = ValidationLevel.STANDARD,
                 significance_threshold: float = 0.05,
                 min_effect_size: float = 0.2,
                 min_power: float = 0.8):
        
        self.validation_level = validation_level
        self.significance_threshold = significance_threshold
        self.min_effect_size = min_effect_size
        self.min_power = min_power
        
        self.logger = get_global_logger(self.__class__.__name__)
        
        # Experimental tracking
        self.experiments: List[ValidationReport] = []
        self.baseline_results: Dict[str, List[float]] = {}
        
        # Statistical configuration
        self.multiple_comparison_methods = ["bonferroni", "holm", "benjamini_hochberg"]
        self.bootstrap_iterations = 10000
        self.confidence_level = 0.95
        
        # Reproducibility tracking
        self.environment_hashes: Set[str] = set()
        self.reproducibility_runs = 3
        
        # Thread pool for parallel experiments
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
    async def validate_novel_algorithm(self, 
                                     novel_algorithm: Callable,
                                     baseline_algorithms: List[Callable],
                                     test_cases: List[Any],
                                     algorithm_name: str = "novel_algorithm") -> ValidationReport:
        """
        Validate a novel algorithm against established baselines.
        
        Performs comprehensive statistical validation including:
        - Multiple baseline comparisons
        - Effect size analysis
        - Power analysis
        - Reproducibility testing
        - Publication-ready artifacts
        """
        
        self.logger.info(f"Starting validation of {algorithm_name}")
        
        # Design experimental conditions
        conditions = self._design_experimental_conditions(
            novel_algorithm, baseline_algorithms, algorithm_name
        )
        
        # Execute experiments
        results = await self._execute_experiments(conditions, test_cases)
        
        # Perform statistical analysis
        statistical_analyses = await self._perform_statistical_analysis(results)
        
        # Assess reproducibility
        reproducibility_score = await self._assess_reproducibility(
            novel_algorithm, baseline_algorithms[0] if baseline_algorithms else None, test_cases
        )
        
        # Generate publication artifacts
        publication_artifacts = await self._generate_publication_artifacts(
            results, statistical_analyses, algorithm_name
        )
        
        # Assess peer review readiness
        peer_review_readiness = self._assess_peer_review_readiness(
            statistical_analyses, reproducibility_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            statistical_analyses, reproducibility_score, peer_review_readiness
        )
        
        validation_report = ValidationReport(
            experiment_type=ExperimentType.NOVEL_ALGORITHM_VALIDATION,
            validation_level=self.validation_level,
            conditions=conditions,
            results=results,
            statistical_analyses=statistical_analyses,
            reproducibility_score=reproducibility_score,
            peer_review_readiness=peer_review_readiness,
            recommendations=recommendations,
            publication_artifacts=publication_artifacts
        )
        
        self.experiments.append(validation_report)
        
        self.logger.info(f"Validation complete for {algorithm_name} - "
                        f"Peer review readiness: {peer_review_readiness:.2f}")
        
        return validation_report
    
    def _design_experimental_conditions(self, 
                                      novel_algorithm: Callable,
                                      baseline_algorithms: List[Callable],
                                      algorithm_name: str) -> List[ExperimentalCondition]:
        """Design comprehensive experimental conditions."""
        
        conditions = []
        
        # Novel algorithm condition
        conditions.append(ExperimentalCondition(
            name=algorithm_name,
            parameters={"algorithm": novel_algorithm, "type": "novel"},
            description=f"Novel algorithm: {algorithm_name}",
            expected_outcome="improved_performance"
        ))
        
        # Baseline conditions
        for i, baseline in enumerate(baseline_algorithms):
            conditions.append(ExperimentalCondition(
                name=f"baseline_{i+1}",
                parameters={"algorithm": baseline, "type": "baseline"},
                description=f"Baseline algorithm {i+1}",
                expected_outcome="baseline_performance"
            ))
        
        # Add control conditions if rigorous validation
        if self.validation_level in [ValidationLevel.RIGOROUS, ValidationLevel.PUBLICATION_READY]:
            conditions.append(ExperimentalCondition(
                name="random_control",
                parameters={"algorithm": self._random_control_algorithm, "type": "control"},
                description="Random control algorithm",
                expected_outcome="poor_performance"
            ))
        
        return conditions
    
    async def _execute_experiments(self, 
                                 conditions: List[ExperimentalCondition],
                                 test_cases: List[Any]) -> List[ExperimentalResult]:
        """Execute experiments across all conditions."""
        
        results = []
        
        # Calculate required sample size
        sample_size = self._calculate_required_sample_size()
        
        for condition in conditions:
            self.logger.info(f"Executing experiments for condition: {condition.name}")
            
            measurements = []
            algorithm = condition.parameters["algorithm"]
            
            # Run multiple trials
            for trial in range(sample_size):
                trial_measurements = []
                
                for test_case in test_cases:
                    # Measure performance
                    start_time = time.perf_counter()
                    
                    try:
                        result = await self._run_algorithm_safely(algorithm, test_case)
                        execution_time = time.perf_counter() - start_time
                        
                        # Calculate performance metric (e.g., execution time, accuracy, etc.)
                        metric = self._calculate_performance_metric(result, execution_time)
                        trial_measurements.append(metric)
                        
                    except Exception as e:
                        self.logger.warning(f"Trial failed for {condition.name}: {e}")
                        trial_measurements.append(float('inf'))  # Penalty for failure
                
                # Aggregate trial measurements
                if trial_measurements:
                    aggregated_measurement = np.mean(trial_measurements)
                    measurements.append(aggregated_measurement)
            
            # Create experimental result
            environment_hash = self._get_environment_hash()
            self.environment_hashes.add(environment_hash)
            
            result = ExperimentalResult(
                condition=condition,
                measurements=measurements,
                metadata={
                    "sample_size": sample_size,
                    "test_cases_count": len(test_cases),
                    "trials_per_condition": sample_size
                },
                timestamp=time.time(),
                environment_hash=environment_hash,
                success=len(measurements) > 0 and not all(np.isinf(measurements))
            )
            
            results.append(result)
        
        return results
    
    def _calculate_required_sample_size(self) -> int:
        """Calculate required sample size for adequate statistical power."""
        
        # Sample size calculation based on desired power and effect size
        alpha = self.significance_threshold
        power = self.min_power
        effect_size = self.min_effect_size
        
        # Using Cohen's formula for t-test sample size
        # n ≈ 2 * (z_α/2 + z_β)² / δ²
        z_alpha = scipy.stats.norm.ppf(1 - alpha/2)
        z_beta = scipy.stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        
        # Minimum sample size based on validation level
        min_samples = {
            ValidationLevel.BASIC: 5,
            ValidationLevel.STANDARD: 10,
            ValidationLevel.RIGOROUS: 20,
            ValidationLevel.PUBLICATION_READY: 30
        }
        
        return max(int(np.ceil(n)), min_samples[self.validation_level])
    
    async def _run_algorithm_safely(self, algorithm: Callable, test_case: Any) -> Any:
        """Run algorithm with error handling."""
        
        @robust_execution(max_retries=2, backoff_factor=1.0)
        async def _safe_run():
            if asyncio.iscoroutinefunction(algorithm):
                return await algorithm(test_case)
            else:
                # Run in thread pool for CPU-bound algorithms
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self.thread_pool, algorithm, test_case)
        
        return await _safe_run()
    
    def _calculate_performance_metric(self, result: Any, execution_time: float) -> float:
        """Calculate performance metric from algorithm result."""
        
        # Default metric is execution time (lower is better)
        # In practice, this would be customized based on the specific research question
        
        if isinstance(result, dict) and "performance" in result:
            return result["performance"]
        elif isinstance(result, (int, float)):
            return float(result)
        else:
            # Use execution time as fallback metric
            return execution_time
    
    def _get_environment_hash(self) -> str:
        """Generate hash of current execution environment."""
        
        environment_info = {
            "python_version": subprocess.check_output(["python3", "--version"], text=True).strip(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "timestamp": int(time.time() / 3600),  # Hour granularity
        }
        
        env_string = json.dumps(environment_info, sort_keys=True)
        return hashlib.md5(env_string.encode()).hexdigest()[:8]
    
    async def _perform_statistical_analysis(self, 
                                          results: List[ExperimentalResult]) -> List[StatisticalAnalysis]:
        """Perform comprehensive statistical analysis."""
        
        analyses = []
        
        # Find novel algorithm and baseline results
        novel_results = None
        baseline_results = []
        
        for result in results:
            if result.condition.parameters.get("type") == "novel":
                novel_results = result
            elif result.condition.parameters.get("type") == "baseline":
                baseline_results.append(result)
        
        if not novel_results or not baseline_results:
            self.logger.warning("Insufficient results for statistical analysis")
            return analyses
        
        # Compare novel algorithm against each baseline
        for baseline_result in baseline_results:
            analysis = await self._compare_conditions(novel_results, baseline_result)
            analyses.append(analysis)
        
        # Apply multiple comparison corrections
        if len(analyses) > 1:
            analyses = self._apply_multiple_comparison_correction(analyses)
        
        # Additional analyses for rigorous validation
        if self.validation_level in [ValidationLevel.RIGOROUS, ValidationLevel.PUBLICATION_READY]:
            
            # Effect size analysis
            for analysis in analyses:
                analysis.effect_size = self._calculate_effect_size(
                    novel_results.measurements, 
                    baseline_results[0].measurements  # Use first baseline for effect size
                )
            
            # Power analysis
            for analysis in analyses:
                analysis.power = self._calculate_statistical_power(
                    novel_results.measurements,
                    baseline_results[0].measurements,
                    analysis.test_statistic
                )
        
        return analyses
    
    async def _compare_conditions(self, 
                                condition1: ExperimentalResult,
                                condition2: ExperimentalResult) -> StatisticalAnalysis:
        """Compare two experimental conditions statistically."""
        
        data1 = np.array(condition1.measurements)
        data2 = np.array(condition2.measurements)
        
        # Remove invalid measurements
        data1 = data1[np.isfinite(data1)]
        data2 = data2[np.isfinite(data2)]
        
        if len(data1) < 3 or len(data2) < 3:
            # Insufficient data
            return StatisticalAnalysis(
                test_type=StatisticalTest.T_TEST,
                test_statistic=0.0,
                p_value=1.0,
                confidence_interval=(0.0, 0.0),
                effect_size=0.0,
                power=0.0,
                sample_size=len(data1) + len(data2),
                significant=False
            )
        
        # Choose appropriate statistical test
        test_type = self._choose_statistical_test(data1, data2)
        
        # Perform the test
        if test_type == StatisticalTest.T_TEST:
            statistic, p_value = scipy.stats.ttest_ind(data1, data2, equal_var=True)
        elif test_type == StatisticalTest.WELCH_T_TEST:
            statistic, p_value = scipy.stats.ttest_ind(data1, data2, equal_var=False)
        elif test_type == StatisticalTest.MANN_WHITNEY_U:
            statistic, p_value = scipy.stats.mannwhitneyu(data1, data2, alternative='two-sided')
        elif test_type == StatisticalTest.BOOTSTRAP:
            statistic, p_value = self._bootstrap_test(data1, data2)
        else:
            # Default to Welch's t-test
            statistic, p_value = scipy.stats.ttest_ind(data1, data2, equal_var=False)
            test_type = StatisticalTest.WELCH_T_TEST
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(data1, data2)
        
        # Effect size
        effect_size = self._calculate_effect_size(data1, data2)
        
        # Statistical power
        power = self._calculate_statistical_power(data1, data2, statistic)
        
        return StatisticalAnalysis(
            test_type=test_type,
            test_statistic=statistic,
            p_value=p_value,
            confidence_interval=confidence_interval,
            effect_size=effect_size,
            power=power,
            sample_size=len(data1) + len(data2),
            significant=p_value < self.significance_threshold
        )
    
    def _choose_statistical_test(self, data1: np.ndarray, data2: np.ndarray) -> StatisticalTest:
        """Choose appropriate statistical test based on data properties."""
        
        # Test for normality
        _, p1 = scipy.stats.shapiro(data1)
        _, p2 = scipy.stats.shapiro(data2)
        
        both_normal = p1 > 0.05 and p2 > 0.05
        
        # Test for equal variances
        _, p_var = scipy.stats.levene(data1, data2)
        equal_variances = p_var > 0.05
        
        if both_normal and equal_variances:
            return StatisticalTest.T_TEST
        elif both_normal and not equal_variances:
            return StatisticalTest.WELCH_T_TEST
        else:
            # Non-parametric test for non-normal data
            return StatisticalTest.MANN_WHITNEY_U
    
    def _bootstrap_test(self, data1: np.ndarray, data2: np.ndarray) -> Tuple[float, float]:
        """Perform bootstrap hypothesis test."""
        
        observed_diff = np.mean(data1) - np.mean(data2)
        
        # Pool the data under null hypothesis
        pooled_data = np.concatenate([data1, data2])
        n1, n2 = len(data1), len(data2)
        
        bootstrap_diffs = []
        for _ in range(self.bootstrap_iterations):
            # Resample under null hypothesis
            resampled = np.random.choice(pooled_data, size=n1 + n2, replace=True)
            sample1 = resampled[:n1]
            sample2 = resampled[n1:]
            
            bootstrap_diff = np.mean(sample1) - np.mean(sample2)
            bootstrap_diffs.append(bootstrap_diff)
        
        bootstrap_diffs = np.array(bootstrap_diffs)
        
        # Two-tailed p-value
        p_value = 2 * min(
            np.mean(bootstrap_diffs >= observed_diff),
            np.mean(bootstrap_diffs <= observed_diff)
        )
        
        return observed_diff, p_value
    
    def _calculate_confidence_interval(self, 
                                     data1: np.ndarray, 
                                     data2: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for the difference in means."""
        
        mean_diff = np.mean(data1) - np.mean(data2)
        
        # Pooled standard error
        n1, n2 = len(data1), len(data2)
        pooled_var = ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2)
        pooled_se = np.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # t-critical value
        df = n1 + n2 - 2
        alpha = 1 - self.confidence_level
        t_critical = scipy.stats.t.ppf(1 - alpha/2, df)
        
        margin_error = t_critical * pooled_se
        
        return (mean_diff - margin_error, mean_diff + margin_error)
    
    def _calculate_effect_size(self, data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        
        mean1, mean2 = np.mean(data1), np.mean(data2)
        n1, n2 = len(data1), len(data2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (mean1 - mean2) / pooled_std
        return abs(cohens_d)
    
    def _calculate_statistical_power(self, 
                                   data1: np.ndarray, 
                                   data2: np.ndarray,
                                   test_statistic: float) -> float:
        """Calculate statistical power of the test."""
        
        # Simplified power calculation
        # In practice, this would use more sophisticated methods
        
        effect_size = self._calculate_effect_size(data1, data2)
        n = (len(data1) + len(data2)) / 2  # Average sample size
        
        # Approximate power calculation using effect size and sample size
        delta = effect_size * np.sqrt(n / 2)
        power = 1 - scipy.stats.norm.cdf(scipy.stats.norm.ppf(1 - self.significance_threshold/2) - delta)
        
        return min(1.0, max(0.0, power))
    
    def _apply_multiple_comparison_correction(self, 
                                            analyses: List[StatisticalAnalysis]) -> List[StatisticalAnalysis]:
        """Apply multiple comparison corrections."""
        
        p_values = [analysis.p_value for analysis in analyses]
        
        # Benjamini-Hochberg correction (most common)
        corrected_p_values = self._benjamini_hochberg_correction(p_values)
        
        for i, analysis in enumerate(analyses):
            analysis.corrected_p_value = corrected_p_values[i]
            analysis.significant = corrected_p_values[i] < self.significance_threshold
        
        return analyses
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg correction for multiple comparisons."""
        
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        # Apply correction
        corrected = np.zeros(n)
        for i in range(n-1, -1, -1):
            rank = i + 1
            corrected[sorted_indices[i]] = min(1.0, sorted_p_values[i] * n / rank)
            
            # Ensure monotonicity
            if i < n - 1:
                corrected[sorted_indices[i]] = min(corrected[sorted_indices[i]], 
                                                 corrected[sorted_indices[i+1]])
        
        return corrected.tolist()
    
    async def _assess_reproducibility(self, 
                                    novel_algorithm: Callable,
                                    baseline_algorithm: Optional[Callable],
                                    test_cases: List[Any]) -> float:
        """Assess reproducibility across different environments/runs."""
        
        reproducibility_scores = []
        
        for run in range(self.reproducibility_runs):
            self.logger.info(f"Reproducibility run {run + 1}/{self.reproducibility_runs}")
            
            # Run novel algorithm
            novel_measurements = []
            for test_case in test_cases:
                start_time = time.perf_counter()
                try:
                    result = await self._run_algorithm_safely(novel_algorithm, test_case)
                    execution_time = time.perf_counter() - start_time
                    metric = self._calculate_performance_metric(result, execution_time)
                    novel_measurements.append(metric)
                except Exception:
                    novel_measurements.append(float('inf'))
            
            # Compare with baseline if available
            if baseline_algorithm:
                baseline_measurements = []
                for test_case in test_cases:
                    start_time = time.perf_counter()
                    try:
                        result = await self._run_algorithm_safely(baseline_algorithm, test_case)
                        execution_time = time.perf_counter() - start_time
                        metric = self._calculate_performance_metric(result, execution_time)
                        baseline_measurements.append(metric)
                    except Exception:
                        baseline_measurements.append(float('inf'))
                
                # Calculate consistency score
                if baseline_measurements and novel_measurements:
                    novel_mean = np.mean([m for m in novel_measurements if np.isfinite(m)])
                    baseline_mean = np.mean([m for m in baseline_measurements if np.isfinite(m)])
                    
                    if baseline_mean > 0:
                        consistency = min(1.0, novel_mean / baseline_mean)
                        reproducibility_scores.append(consistency)
        
        if reproducibility_scores:
            # Calculate coefficient of variation as reproducibility measure
            mean_score = np.mean(reproducibility_scores)
            std_score = np.std(reproducibility_scores)
            cv = std_score / mean_score if mean_score > 0 else 1.0
            
            # Convert to reproducibility score (lower CV = higher reproducibility)
            reproducibility = max(0.0, 1.0 - cv)
        else:
            reproducibility = 0.0
        
        return reproducibility
    
    async def _generate_publication_artifacts(self, 
                                            results: List[ExperimentalResult],
                                            analyses: List[StatisticalAnalysis],
                                            algorithm_name: str) -> Dict[str, Any]:
        """Generate publication-ready artifacts."""
        
        artifacts = {}
        
        # Create results table
        results_df = self._create_results_table(results, analyses)
        artifacts["results_table"] = results_df.to_dict()
        
        # Generate plots
        if self.validation_level == ValidationLevel.PUBLICATION_READY:
            plots = await self._generate_publication_plots(results, analyses, algorithm_name)
            artifacts["plots"] = plots
        
        # Create statistical summary
        artifacts["statistical_summary"] = self._create_statistical_summary(analyses)
        
        # Generate LaTeX tables
        artifacts["latex_table"] = self._generate_latex_table(results_df)
        
        # Create experimental protocol
        artifacts["experimental_protocol"] = self._create_experimental_protocol(results)
        
        return artifacts
    
    def _create_results_table(self, 
                            results: List[ExperimentalResult],
                            analyses: List[StatisticalAnalysis]) -> pd.DataFrame:
        """Create comprehensive results table."""
        
        table_data = []
        
        for result in results:
            measurements = [m for m in result.measurements if np.isfinite(m)]
            
            row = {
                "Algorithm": result.condition.name,
                "Mean": np.mean(measurements) if measurements else np.nan,
                "Std": np.std(measurements, ddof=1) if len(measurements) > 1 else np.nan,
                "Min": np.min(measurements) if measurements else np.nan,
                "Max": np.max(measurements) if measurements else np.nan,
                "Sample_Size": len(measurements),
                "Success_Rate": len(measurements) / len(result.measurements)
            }
            table_data.append(row)
        
        # Add statistical test results
        for i, analysis in enumerate(analyses):
            if i < len(table_data) - 1:  # Avoid index error
                table_data[i+1][f"p_value_vs_novel"] = analysis.p_value
                table_data[i+1][f"effect_size"] = analysis.effect_size
                table_data[i+1][f"significant"] = analysis.significant
        
        return pd.DataFrame(table_data)
    
    async def _generate_publication_plots(self, 
                                        results: List[ExperimentalResult],
                                        analyses: List[StatisticalAnalysis],
                                        algorithm_name: str) -> Dict[str, str]:
        """Generate publication-quality plots."""
        
        plots = {}
        
        # Set publication style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Performance comparison plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        algorithm_names = []
        performance_data = []
        
        for result in results:
            measurements = [m for m in result.measurements if np.isfinite(m)]
            if measurements:
                algorithm_names.append(result.condition.name)
                performance_data.append(measurements)
        
        if performance_data:
            # Box plot
            box_plot = ax.boxplot(performance_data, labels=algorithm_names, patch_artist=True)
            
            # Color the boxes
            colors = sns.color_palette("husl", len(box_plot['boxes']))
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel('Performance Metric')
            ax.set_title(f'Performance Comparison: {algorithm_name}')
            ax.grid(True, alpha=0.3)
            
            # Add significance annotations
            self._add_significance_annotations(ax, analyses, performance_data)
            
            # Save plot
            plot_path = f"/tmp/{algorithm_name}_performance_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots["performance_comparison"] = plot_path
            plt.close()
        
        # Effect size plot
        if analyses and any(analysis.effect_size > 0 for analysis in analyses):
            fig, ax = plt.subplots(figsize=(8, 6))
            
            effect_sizes = [analysis.effect_size for analysis in analyses]
            comparison_names = [f"vs Baseline {i+1}" for i in range(len(analyses))]
            
            bars = ax.bar(comparison_names, effect_sizes, color=sns.color_palette("viridis", len(effect_sizes)))
            
            # Add effect size interpretation lines
            ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect')
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Medium effect')
            ax.axhline(y=0.8, color='darkred', linestyle='--', alpha=0.7, label='Large effect')
            
            ax.set_ylabel("Cohen's d (Effect Size)")
            ax.set_title(f"Effect Size Analysis: {algorithm_name}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = f"/tmp/{algorithm_name}_effect_sizes.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plots["effect_sizes"] = plot_path
            plt.close()
        
        return plots
    
    def _add_significance_annotations(self, 
                                    ax: plt.Axes, 
                                    analyses: List[StatisticalAnalysis],
                                    performance_data: List[List[float]]):
        """Add statistical significance annotations to plot."""
        
        if len(performance_data) < 2:
            return
        
        y_max = max(max(data) for data in performance_data if data)
        y_range = y_max - min(min(data) for data in performance_data if data)
        
        for i, analysis in enumerate(analyses):
            if analysis.significant:
                # Add significance line
                y_sig = y_max + 0.1 * y_range * (i + 1)
                
                ax.plot([0, i + 1], [y_sig, y_sig], 'k-', linewidth=1)
                ax.plot([0, 0], [y_sig - 0.02 * y_range, y_sig], 'k-', linewidth=1)
                ax.plot([i + 1, i + 1], [y_sig - 0.02 * y_range, y_sig], 'k-', linewidth=1)
                
                # Add significance marker
                p_val = analysis.corrected_p_value if analysis.corrected_p_value else analysis.p_value
                if p_val < 0.001:
                    sig_text = "***"
                elif p_val < 0.01:
                    sig_text = "**"
                elif p_val < 0.05:
                    sig_text = "*"
                else:
                    sig_text = "ns"
                
                ax.text((0 + i + 1) / 2, y_sig + 0.01 * y_range, sig_text, 
                       ha='center', va='bottom', fontweight='bold')
    
    def _create_statistical_summary(self, analyses: List[StatisticalAnalysis]) -> Dict[str, Any]:
        """Create statistical summary."""
        
        return {
            "total_comparisons": len(analyses),
            "significant_comparisons": sum(1 for a in analyses if a.significant),
            "average_effect_size": np.mean([a.effect_size for a in analyses]) if analyses else 0.0,
            "average_power": np.mean([a.power for a in analyses]) if analyses else 0.0,
            "multiple_comparison_corrected": any(a.corrected_p_value is not None for a in analyses),
            "tests_used": list(set(a.test_type.value for a in analyses))
        }
    
    def _generate_latex_table(self, results_df: pd.DataFrame) -> str:
        """Generate LaTeX table for publication."""
        
        # Format numeric columns
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in ['p_value_vs_novel']:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            elif col in ['Mean', 'Std', 'effect_size']:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
            else:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        
        # Generate LaTeX
        latex_table = results_df.to_latex(
            index=False,
            escape=False,
            column_format='l' + 'c' * (len(results_df.columns) - 1),
            caption="Experimental Results and Statistical Analysis",
            label="tab:results"
        )
        
        return latex_table
    
    def _create_experimental_protocol(self, results: List[ExperimentalResult]) -> Dict[str, Any]:
        """Create detailed experimental protocol."""
        
        return {
            "experimental_design": {
                "type": "randomized_controlled_trial",
                "conditions": len(results),
                "sample_size_per_condition": results[0].metadata.get("sample_size", 0) if results else 0,
                "total_measurements": sum(len(r.measurements) for r in results)
            },
            "data_collection": {
                "measurement_type": "performance_metric",
                "environment_controls": list(self.environment_hashes),
                "reproducibility_runs": self.reproducibility_runs
            },
            "statistical_analysis": {
                "significance_threshold": self.significance_threshold,
                "confidence_level": self.confidence_level,
                "multiple_comparison_correction": "benjamini_hochberg",
                "effect_size_measure": "cohens_d"
            }
        }
    
    def _assess_peer_review_readiness(self, 
                                    analyses: List[StatisticalAnalysis],
                                    reproducibility_score: float) -> float:
        """Assess readiness for peer review."""
        
        readiness_score = 0.0
        
        # Statistical rigor (40% weight)
        if analyses:
            significant_results = sum(1 for a in analyses if a.significant)
            statistical_rigor = significant_results / len(analyses)
            
            # Check for adequate power
            adequate_power = sum(1 for a in analyses if a.power >= self.min_power) / len(analyses)
            
            # Check for adequate effect sizes
            adequate_effect = sum(1 for a in analyses if a.effect_size >= self.min_effect_size) / len(analyses)
            
            statistical_score = (statistical_rigor + adequate_power + adequate_effect) / 3
            readiness_score += statistical_score * 0.4
        
        # Reproducibility (30% weight)
        readiness_score += reproducibility_score * 0.3
        
        # Sample size adequacy (20% weight)
        if analyses:
            avg_sample_size = np.mean([a.sample_size for a in analyses])
            sample_adequacy = min(1.0, avg_sample_size / 30)  # 30 as ideal minimum
            readiness_score += sample_adequacy * 0.2
        
        # Validation level (10% weight)
        level_scores = {
            ValidationLevel.BASIC: 0.25,
            ValidationLevel.STANDARD: 0.5,
            ValidationLevel.RIGOROUS: 0.75,
            ValidationLevel.PUBLICATION_READY: 1.0
        }
        readiness_score += level_scores[self.validation_level] * 0.1
        
        return min(1.0, readiness_score)
    
    def _generate_recommendations(self, 
                                analyses: List[StatisticalAnalysis],
                                reproducibility_score: float,
                                peer_review_readiness: float) -> List[str]:
        """Generate recommendations for improving research quality."""
        
        recommendations = []
        
        # Statistical recommendations
        if analyses:
            low_power_analyses = [a for a in analyses if a.power < self.min_power]
            if low_power_analyses:
                recommendations.append(
                    f"Increase sample size for {len(low_power_analyses)} comparisons "
                    f"to achieve adequate statistical power (≥{self.min_power})"
                )
            
            small_effect_analyses = [a for a in analyses if a.effect_size < self.min_effect_size]
            if small_effect_analyses:
                recommendations.append(
                    f"Consider practical significance: {len(small_effect_analyses)} comparisons "
                    f"show statistically significant but small effect sizes"
                )
            
            non_significant = [a for a in analyses if not a.significant]
            if non_significant:
                recommendations.append(
                    f"Investigate {len(non_significant)} non-significant comparisons: "
                    "consider alternative algorithms or parameter tuning"
                )
        
        # Reproducibility recommendations
        if reproducibility_score < 0.8:
            recommendations.append(
                f"Improve reproducibility (current: {reproducibility_score:.2f}): "
                "ensure deterministic algorithms and controlled experimental conditions"
            )
        
        # Peer review recommendations
        if peer_review_readiness < 0.7:
            recommendations.append(
                f"Enhance peer review readiness (current: {peer_review_readiness:.2f}): "
                "consider increasing validation level and sample sizes"
            )
        
        # Validation level recommendations
        if self.validation_level == ValidationLevel.BASIC:
            recommendations.append(
                "Consider upgrading to STANDARD or RIGOROUS validation level "
                "for stronger scientific evidence"
            )
        
        return recommendations
    
    def _random_control_algorithm(self, test_case: Any) -> Any:
        """Random control algorithm for experimental control."""
        # Return random result
        return {"performance": np.random.exponential(10.0)}
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation experiments."""
        
        if not self.experiments:
            return {"message": "No experiments completed"}
        
        return {
            "total_experiments": len(self.experiments),
            "validation_levels": [exp.validation_level.value for exp in self.experiments],
            "average_peer_review_readiness": np.mean([exp.peer_review_readiness for exp in self.experiments]),
            "average_reproducibility": np.mean([exp.reproducibility_score for exp in self.experiments]),
            "total_statistical_tests": sum(len(exp.statistical_analyses) for exp in self.experiments),
            "significant_results": sum(
                sum(1 for analysis in exp.statistical_analyses if analysis.significant)
                for exp in self.experiments
            )
        }