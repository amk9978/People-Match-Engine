#!/usr/bin/env python3
import asyncio
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from graph_matcher import GraphMatcher
from statistical_analysis import FeatureAnalyzer, SubgraphAnalyzer


class DataEngineer:
    """Data engineering pipeline for feature analysis and optimization"""

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.feature_analyzer = None
        self.subgraph_analyzer = None
        self.analysis_results = {}

    def load_and_analyze_data(self) -> Dict:
        """Load data and run comprehensive feature analysis"""
        # Load data
        self.df = pd.read_csv(self.csv_path)
        self.feature_analyzer = FeatureAnalyzer(self.df)

        print(f"Loaded {len(self.df)} records from {self.csv_path}")

        # Basic data quality analysis
        quality_metrics = self._analyze_data_quality()

        # Tag frequency and dominance analysis
        print("Analyzing tag frequency distributions...")
        tag_analysis = self.feature_analyzer.analyze_tag_frequency_distribution()

        self.analysis_results = {
            "data_quality": quality_metrics,
            "tag_analysis": tag_analysis,
        }

        return self.analysis_results

    def analyze_feature_importance(
        self,
        feature_embeddings: Dict[str, np.ndarray],
        feature_weights: Dict[str, float],
    ) -> Dict:
        """Comprehensive feature importance analysis"""
        print("Running feature importance analysis...")

        # Variance analysis
        print("  - Calculating variance metrics...")
        variance_analysis = self.feature_analyzer.calculate_feature_variance(
            feature_embeddings
        )

        # Correlation analysis
        print("  - Calculating feature correlations...")
        correlation_analysis = self.feature_analyzer.calculate_feature_correlations(
            feature_embeddings
        )

        # Feature ablation analysis
        print("  - Running feature ablation analysis...")
        ablation_analysis = self.feature_analyzer.feature_ablation_analysis(
            feature_embeddings, feature_weights
        )

        importance_results = {
            "variance_analysis": variance_analysis,
            "correlation_analysis": correlation_analysis,
            "ablation_analysis": ablation_analysis,
        }

        self.analysis_results["feature_importance"] = importance_results
        return importance_results

    def analyze_subgraph_quality(
        self, graph, feature_embeddings: Dict[str, np.ndarray], subgraph_nodes: Set[int]
    ) -> Dict:
        """Comprehensive subgraph quality analysis"""
        print("Running subgraph quality analysis...")

        self.subgraph_analyzer = SubgraphAnalyzer(graph, feature_embeddings)

        # Comprehensive subgraph analysis
        quality_results = self.subgraph_analyzer.comprehensive_subgraph_analysis(
            subgraph_nodes
        )

        self.analysis_results["subgraph_quality"] = quality_results
        return quality_results

    def _analyze_data_quality(self) -> Dict:
        """Analyze basic data quality metrics"""
        quality_metrics = {}

        for feature_name, column_name in self.feature_analyzer.feature_columns.items():
            column_data = self.df[column_name]

            # Missing value analysis
            missing_count = column_data.isna().sum()
            missing_ratio = missing_count / len(self.df)

            # Empty string analysis
            empty_count = sum(
                1 for val in column_data if pd.notna(val) and str(val).strip() == ""
            )
            empty_ratio = empty_count / len(self.df)

            # Length distribution
            lengths = [len(str(val)) if pd.notna(val) else 0 for val in column_data]

            quality_metrics[feature_name] = {
                "missing_count": int(missing_count),
                "missing_ratio": float(missing_ratio),
                "empty_count": int(empty_count),
                "empty_ratio": float(empty_ratio),
                "mean_length": float(np.mean(lengths)),
                "median_length": float(np.median(lengths)),
                "min_length": int(np.min(lengths)),
                "max_length": int(np.max(lengths)),
                "data_completeness": float(1 - missing_ratio - empty_ratio),
            }

        return quality_metrics

    def generate_feature_ranking(self) -> Dict[str, Dict]:
        """Generate comprehensive feature ranking based on all analyses"""
        if "feature_importance" not in self.analysis_results:
            return {}

        rankings = {}
        importance_data = self.analysis_results["feature_importance"]

        # Ranking based on different criteria
        for feature_name in self.feature_analyzer.feature_columns.keys():

            # Variance-based score (higher variance = more discriminative)
            variance_score = 0
            if feature_name in importance_data["variance_analysis"]:
                var_data = importance_data["variance_analysis"][feature_name]
                variance_score = (
                    var_data["mean_pairwise_distance"] * var_data["distance_range"]
                )

            # Ablation-based score (higher importance = more critical)
            ablation_score = 0
            if feature_name in importance_data["ablation_analysis"]:
                ablation_score = importance_data["ablation_analysis"][feature_name][
                    "importance_score"
                ]

            # Tag quality score
            tag_score = 0
            if (
                "tag_analysis" in self.analysis_results
                and feature_name in self.analysis_results["tag_analysis"]
            ):
                tag_data = self.analysis_results["tag_analysis"][feature_name]
                # Higher unique tags and lower dominance = better discrimination
                tag_score = (
                    tag_data["unique_tag_count"]
                    * (1 - tag_data["gini_coefficient"])
                    * (1 - tag_data["singleton_tag_ratio"])
                )

            # Data quality score
            quality_score = 0
            if (
                "data_quality" in self.analysis_results
                and feature_name in self.analysis_results["data_quality"]
            ):
                quality_data = self.analysis_results["data_quality"][feature_name]
                quality_score = quality_data["data_completeness"]

            # Combined score (normalize and weight)
            rankings[feature_name] = {
                "variance_score": float(variance_score),
                "ablation_score": float(ablation_score),
                "tag_quality_score": float(tag_score),
                "data_quality_score": float(quality_score),
                "combined_score": float(
                    0.3 * variance_score
                    + 0.4 * ablation_score
                    + 0.2 * tag_score
                    + 0.1 * quality_score
                ),
            }

        # Sort by combined score
        sorted_features = sorted(
            rankings.items(), key=lambda x: x[1]["combined_score"], reverse=True
        )

        return {
            "rankings": rankings,
            "sorted_features": sorted_features,
            "recommendations": self._generate_feature_recommendations(sorted_features),
        }

    def _generate_feature_recommendations(self, sorted_features: List[Tuple]) -> Dict:
        """Generate actionable recommendations based on feature analysis"""
        recommendations = {
            "keep_features": [],
            "consider_removing": [],
            "needs_improvement": [],
            "analysis_summary": {},
        }

        for feature_name, scores in sorted_features:
            combined_score = scores["combined_score"]

            if combined_score > 0.6:
                recommendations["keep_features"].append(
                    {
                        "feature": feature_name,
                        "score": combined_score,
                        "reason": "High discriminative power and data quality",
                    }
                )
            elif combined_score < 0.3:
                recommendations["consider_removing"].append(
                    {
                        "feature": feature_name,
                        "score": combined_score,
                        "reason": "Low discriminative power or poor data quality",
                    }
                )
            else:
                recommendations["needs_improvement"].append(
                    {
                        "feature": feature_name,
                        "score": combined_score,
                        "reason": "Moderate performance, potential for improvement",
                    }
                )

        # Summary statistics
        scores_only = [scores["combined_score"] for _, scores in sorted_features]
        recommendations["analysis_summary"] = {
            "mean_feature_score": float(np.mean(scores_only)),
            "std_feature_score": float(np.std(scores_only)),
            "best_feature": sorted_features[0][0] if sorted_features else None,
            "worst_feature": sorted_features[-1][0] if sorted_features else None,
            "score_range": (
                float(max(scores_only) - min(scores_only)) if scores_only else 0.0
            ),
        }

        return recommendations

    def export_analysis_report(self, output_path: str = None) -> str:
        """Export comprehensive analysis report"""
        if not output_path:
            output_path = self.csv_path.replace(".csv", "_analysis_report.json")

        import json

        with open(output_path, "w") as f:
            json.dump(self.analysis_results, f, indent=2, default=str)

        print(f"Analysis report exported to: {output_path}")
        return output_path

    def print_analysis_summary(self):
        """Print human-readable analysis summary"""
        if not self.analysis_results:
            print("No analysis results available. Run analysis first.")
            return

        print("\n" + "=" * 60)
        print("DATA ENGINEERING ANALYSIS SUMMARY")
        print("=" * 60)

        # Data Quality Summary
        if "data_quality" in self.analysis_results:
            print("\n📊 DATA QUALITY OVERVIEW:")
            for feature, metrics in self.analysis_results["data_quality"].items():
                print(
                    f"  {feature:15}: {metrics['data_completeness']:.1%} complete, "
                    f"avg length {metrics['mean_length']:.0f}"
                )

        # Tag Analysis Summary
        if "tag_analysis" in self.analysis_results:
            print("\n🏷️  TAG DISTRIBUTION ANALYSIS:")
            for feature, metrics in self.analysis_results["tag_analysis"].items():
                print(
                    f"  {feature:15}: {metrics['unique_tag_count']:3d} unique tags, "
                    f"Gini: {metrics['gini_coefficient']:.2f}, "
                    f"Top-5: {metrics['top_5_tag_coverage']:.1%}"
                )

        # Feature Rankings
        if "feature_importance" in self.analysis_results:
            rankings = self.generate_feature_ranking()
            if "sorted_features" in rankings:
                print("\n🎯 FEATURE IMPORTANCE RANKING:")
                for i, (feature, scores) in enumerate(rankings["sorted_features"], 1):
                    print(
                        f"  {i}. {feature:15}: {scores['combined_score']:.3f} "
                        f"(Var: {scores['variance_score']:.2f}, "
                        f"Abl: {scores['ablation_score']:.2f})"
                    )

        # Subgraph Quality
        if "subgraph_quality" in self.analysis_results:
            quality = self.analysis_results["subgraph_quality"]
            print(f"\n🎯 SUBGRAPH QUALITY METRICS:")
            print(
                f"  Size: {quality['size_metrics']['subgraph_size']} nodes "
                f"({quality['size_metrics']['coverage_ratio']:.1%} coverage)"
            )

            if "density_analysis" in quality:
                density = quality["density_analysis"]
                print(
                    f"  Density: Internal {density['internal_density']:.3f}, "
                    f"Boundary {density['boundary_density']:.3f}, "
                    f"Gradient {density['density_gradient']:.3f}"
                )


async def run_comprehensive_analysis(csv_path: str, min_density: float = 0.1) -> Dict:
    """Run complete data engineering analysis pipeline"""
    print("Starting comprehensive data engineering analysis...")

    # Initialize data engineer
    engineer = DataEngineer(csv_path)

    # Step 1: Load and analyze data quality
    basic_analysis = engineer.load_and_analyze_data()

    # Step 2: Run graph matching to get embeddings
    print("\nRunning GraphMatcher to generate embeddings...")
    matcher = GraphMatcher(csv_path, min_density)
    matcher.load_data()
    feature_embeddings = await matcher.embed_features()
    matcher.create_graph(feature_embeddings)

    # Step 3: Feature importance analysis
    feature_weights = {
        "role_spec": 0.25,
        "experience": 0.15,
        "industry": 0.20,
        "market": 0.15,
        "offering": 0.15,
        "personas": 0.10,
    }
    importance_analysis = engineer.analyze_feature_importance(
        feature_embeddings, feature_weights
    )

    # Step 4: Find subgraph and analyze quality
    largest_dense_nodes, density = matcher.find_largest_dense_subgraph()
    subgraph_analysis = engineer.analyze_subgraph_quality(
        matcher.graph, feature_embeddings, largest_dense_nodes
    )

    # Step 5: Generate rankings and recommendations
    rankings = engineer.generate_feature_ranking()
    engineer.analysis_results["feature_rankings"] = rankings

    # Step 6: Print summary
    engineer.print_analysis_summary()

    # Step 7: Export report
    report_path = engineer.export_analysis_report()
    print(engineer.analysis_results)
    return engineer.analysis_results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_analysis("data/test_batch2.csv"))
