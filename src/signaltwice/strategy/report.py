from __future__ import annotations

from typing import Dict, List

from signaltwice.core.interface import BaseReportSection
from signaltwice.core.utils import ReportStyler, SafeExtractor
from signaltwice.engine.registry import reports


@reports.register_strategy("dataOverview")
class DataOverviewStrategy(BaseReportSection):
    def calculate(self, raw_data: dict) -> dict:
        dtypes = SafeExtractor.mapping_from(raw_data, "get_dtypes")

        rows = SafeExtractor.deep_number(raw_data, ["get_shape", "rows"], 0)
        columns = SafeExtractor.deep_number(raw_data, ["get_shape", "columns"], 0)
        duplicate_rows = SafeExtractor.deep_number(
            raw_data, ["check_duplication", "duplicate_rows"], 0
        )

        type_groups: Dict[str, List[str]] = {}
        for col, dtype in dtypes.items():
            dtype_key = str(dtype)
            type_groups.setdefault(dtype_key, []).append(col)

        grouped = []
        for dtype_key, cols in type_groups.items():
            grouped.append(
                {
                    "dtype": dtype_key,
                    "count": len(cols),
                    "columns": cols,
                }
            )

        return {
            "rows": rows,
            "columns": columns,
            "duplicate_rows": duplicate_rows,
            "has_duplicates": duplicate_rows > 0,
            "dtype_groups": grouped,
        }

    def render(self, metrics: dict, styler: ReportStyler) -> str:
        rows = SafeExtractor.number(metrics.get("rows"), 0)
        columns = SafeExtractor.number(metrics.get("columns"), 0)
        duplicate_rows = SafeExtractor.number(metrics.get("duplicate_rows"), 0)
        dtype_groups = SafeExtractor.list(metrics.get("dtype_groups"))

        row_label = styler.count_noun(rows, "row", "rows")
        column_label = styler.count_noun(columns, "column", "columns")
        duplicate_label = styler.count_noun(
            duplicate_rows, "duplicate row", "duplicate rows"
        )

        lines = [styler.header("1. Data Overview (DATA OVERVIEW)")]
        lines.append(styler.key_value(f"Total {row_label}", f"{rows:,}", icon="📊"))
        lines.append(
            styler.key_value(f"Total {column_label}", f"{columns:,}", icon="🧩")
        )

        dupe_color = styler.red() if duplicate_rows > 0 else styler.green()
        dupe_value = f"{duplicate_rows:,}" if duplicate_rows > 0 else "0"
        lines.append(
            styler.key_value(duplicate_label, dupe_value, color=dupe_color, icon="👯")
        )

        lines.append(styler.sub_header("Column Type Distribution"))
        if not dtype_groups:
            lines.append("  (No column info)")
        else:
            for group in dtype_groups:
                dtype = group.get("dtype")
                count = SafeExtractor.number(group.get("count"), 0)
                cols = SafeExtractor.deep_list(group, ["columns"])
                example = ", ".join(cols[:3])
                if len(cols) > 3:
                    example += "..."
                example = SafeExtractor.truncate(example, 30)
                column_label = styler.count_noun(count, "column", "columns")
                lines.append(
                    f"   ▪ {dtype:<10} : {count:>2} {column_label} ({example})"
                )

        return "\n".join(lines)


@reports.register_strategy("missingValueReport")
class MissingValueStrategy(BaseReportSection):
    def calculate(self, raw_data: dict) -> dict:
        null_data = SafeExtractor.mapping_from(raw_data, "count_nulls")
        total_rows = SafeExtractor.deep_number(raw_data, ["get_shape", "rows"], 0)

        empty_rows_count = SafeExtractor.deep_number(
            raw_data, ["check_data_health", "empty_rows_count"], 0
        )
        empty_rows_percent = SafeExtractor.deep_float(
            raw_data, ["check_data_health", "empty_rows_percent"], None
        )
        if empty_rows_percent is None:
            empty_rows_percent = (
                (empty_rows_count / total_rows * 100) if total_rows else 0.0
            )

        missing_columns = []
        for col, stats in null_data.items():
            count = SafeExtractor.deep_number(stats, ["count"], 0)
            if count <= 0:
                continue
            percent = SafeExtractor.deep_float(stats, ["percent"], None)
            if percent is None:
                percent = (count / total_rows * 100) if total_rows else 0.0
            missing_columns.append(
                {
                    "column": col,
                    "count": count,
                    "percent": percent,
                }
            )

        return {
            "empty_rows_count": empty_rows_count,
            "empty_rows_percent": empty_rows_percent,
            "missing_columns": missing_columns,
        }

    def render(self, metrics: dict, styler: ReportStyler) -> str:
        empty_rows_count = SafeExtractor.number(metrics.get("empty_rows_count"), 0)
        empty_rows_percent = SafeExtractor.float(metrics.get("empty_rows_percent"), 0.0)
        missing_columns = SafeExtractor.list(metrics.get("missing_columns"))
        empty_row_label = styler.count_noun(empty_rows_count, "row", "rows")

        lines = [styler.header("2. Missing Values (MISSING VALUES)")]

        if empty_rows_count > 0:
            lines.append(
                styler.error_text(
                    f"Found {empty_rows_count:,} empty {empty_row_label} "
                    "(missing > 50%)"
                )
            )
        else:
            lines.append(
                styler.success_text(
                    "No high-missing rows detected (empty row check passed)"
                )
            )

        lines.append(styler.sub_header("Column Missing Details"))

        if not missing_columns:
            lines.append(f"   {styler.success_text('All columns are complete')}")
        else:
            for item in missing_columns:
                col = item.get("column")
                count = SafeExtractor.number(item.get("count"), 0)
                percent = SafeExtractor.float(item.get("percent"), 0.0)
                row_label = styler.count_noun(count, "row", "rows")
                color = styler.red() if percent > 20 else styler.yellow()
                bar_len = int(max(0, min(100, percent)) / 5)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(
                    f"   {col:<15} |{color}{bar}{styler.reset()}| "
                    f"{percent:>6.2f}% ({count:,} {row_label})"
                )

        if empty_rows_count > 0:
            lines.append(
                styler.key_value(
                    "Empty row percent",
                    f"{empty_rows_percent:.2f}",
                    unit="%",
                )
            )

        return "\n".join(lines)


@reports.register_strategy("numericStats")
class NumericStatsStrategy(BaseReportSection):
    def calculate(self, raw_data: dict) -> dict:
        stats = SafeExtractor.mapping_from(raw_data, "get_descriptive_stats")
        dist = SafeExtractor.mapping_from(raw_data, "calc_distribution_shape")
        outliers = SafeExtractor.mapping_from(raw_data, "detect_outliers")

        columns = []
        for col, col_stats in stats.items():
            stat_map = SafeExtractor.mapping(col_stats)
            skew = SafeExtractor.deep_float(dist, [col, "skewness"], None)
            kurt = SafeExtractor.deep_float(dist, [col, "kurtosis"], None)

            columns.append(
                {
                    "column": col,
                    "mean": SafeExtractor.float(stat_map.get("mean"), 0.0),
                    "std": SafeExtractor.float(stat_map.get("std"), 0.0),
                    "min": stat_map.get("min"),
                    "max": stat_map.get("max"),
                    "skewness": skew,
                    "kurtosis": kurt,
                    "skewness_description": SafeExtractor.describe_skewness(skew),
                    "outlier_count": SafeExtractor.deep_number(
                        outliers, [col, "outlier_count"], 0
                    ),
                }
            )

        return {"columns": columns}

    def render(self, metrics: dict, styler: ReportStyler) -> str:
        columns = SafeExtractor.list(metrics.get("columns"))

        lines = [styler.header("3. Numeric Stats (NUMERICAL STATS)")]

        if not columns:
            lines.append("  (No numeric columns detected)")
            return "\n".join(lines)

        for item in columns:
            col = item.get("column")
            mean_val = SafeExtractor.float(item.get("mean"), 0.0)
            std_val = SafeExtractor.float(item.get("std"), 0.0)
            min_val = item.get("min")
            max_val = item.get("max")
            skew = item.get("skewness")
            kurt = item.get("kurtosis")
            skew_desc = item.get("skewness_description")
            outlier_count = SafeExtractor.number(item.get("outlier_count"), 0)

            lines.append(f"\n{styler.bold()}🔹 {col}{styler.reset()}")
            lines.append(
                "   ├─ 📈 Distribution: "
                f"Mean={mean_val:.2f} | Std={std_val:.2f} | "
                f"Range=[{min_val}, {max_val}]"
            )

            if isinstance(skew, (int, float)):
                desc = f" ({skew_desc})" if skew_desc else ""
                lines.append(
                    f"   ├─ 📐 Shape: Skew={skew:.2f}{desc} | Kurt={kurt:.2f}"
                )

            outlier_text = (
                f"{styler.yellow()}{outlier_count:,} (IQR){styler.reset()}"
                if outlier_count > 0
                else f"{styler.green()}0{styler.reset()}"
            )
            lines.append(f"   └─ 🚨 Outliers: {outlier_text}")

        return "\n".join(lines)


@reports.register_strategy("categoricalStats")
class CategoricalStatsStrategy(BaseReportSection):
    def calculate(self, raw_data: dict) -> dict:
        cardinality = SafeExtractor.mapping_from(raw_data, "count_uniques")
        rare = SafeExtractor.mapping_from(raw_data, "check_rare_categories")
        noise = SafeExtractor.mapping_from(raw_data, "check_string_variants")

        columns = []
        for col, count in cardinality.items():
            rare_count = SafeExtractor.deep_number(rare, [col, "rare_category_count"], 0)
            noise_count = SafeExtractor.deep_number(
                noise, [col, "variant_noise_count"], 0
            )
            columns.append(
                {
                    "column": col,
                    "unique_values": SafeExtractor.number(count, 0),
                    "rare_category_count": rare_count,
                    "variant_noise_count": noise_count,
                }
            )

        return {"columns": columns}

    def render(self, metrics: dict, styler: ReportStyler) -> str:
        columns = SafeExtractor.list(metrics.get("columns"))

        lines = [styler.header("4. Categorical & String Quality (CATEGORICAL)")]

        if not columns:
            lines.append("  (No categorical/string columns detected)")
            return "\n".join(lines)

        for item in columns:
            col = item.get("column")
            display_col = SafeExtractor.truncate(str(col), 30)
            unique_values = SafeExtractor.number(item.get("unique_values"), 0)
            rare_count = SafeExtractor.number(item.get("rare_category_count"), 0)
            noise_count = SafeExtractor.number(item.get("variant_noise_count"), 0)

            lines.append(f"\n{styler.bold()}🔸 {display_col}{styler.reset()}")
            lines.append(f"   ├─ 🏷️  Unique Values: {unique_values:,}")

            if rare_count == 0 and noise_count == 0:
                lines.append(
                    f"   └─ {styler.success_text('No rare categories or string noise')}"
                )
            else:
                issues = []
                if rare_count > 0:
                    issues.append(styler.warning_text(f"Rare categories: {rare_count}"))
                else:
                    issues.append(styler.success_text("Rare categories: None"))

                if noise_count > 0:
                    issues.append(styler.warning_text(f"String noise: {noise_count}"))
                else:
                    issues.append(styler.success_text("String noise: None"))

                lines.append(f"   └─ {' | '.join(issues)}")

        return "\n".join(lines)


@reports.register_strategy("dataIntegrity")
class DataIntegrityStrategy(BaseReportSection):
    def calculate(self, raw_data: dict) -> dict:
        constants = SafeExtractor.mapping_from(raw_data, "check_constant_columns")
        ids = SafeExtractor.mapping_from(raw_data, "check_id_integrity")

        constant_columns = [
            col
            for col, info in constants.items()
            if SafeExtractor.mapping(info).get("is_constant")
        ]

        id_checks = []
        for col, info in ids.items():
            inconsistent = SafeExtractor.deep_number(info, ["inconsistent_row_count"], 0)
            id_checks.append(
                {
                    "column": col,
                    "inconsistent_row_count": inconsistent,
                }
            )

        return {
            "constant_columns": constant_columns,
            "id_integrity": id_checks,
        }

    def render(self, metrics: dict, styler: ReportStyler) -> str:
        constant_columns = SafeExtractor.list(metrics.get("constant_columns"))
        id_checks = SafeExtractor.list(metrics.get("id_integrity"))

        lines = [styler.header("5. Data Integrity (DATA INTEGRITY)")]

        lines.append(styler.sub_header("Constant columns check"))
        if constant_columns:
            column_label = styler.count_noun(
                len(constant_columns), "column", "columns"
            )
            lines.append(
                styler.warning_text(
                    f"Found {len(constant_columns)} constant {column_label} (no variance):"
                )
            )
            lines.append(
                f"      {styler.red()}{', '.join(constant_columns)}{styler.reset()}"
            )
            lines.append("      -> Consider removing them (no predictive value).")
        else:
            lines.append(f"   {styler.success_text('No constant columns found')}")

        lines.append(styler.sub_header("ID integrity check"))
        if not id_checks:
            lines.append("   (No ID columns specified)")
        else:
            issues_found = False
            for item in id_checks:
                col = item.get("column")
                inconsistent = SafeExtractor.number(
                    item.get("inconsistent_row_count"), 0
                )
                if inconsistent > 0:
                    row_label = styler.count_noun(inconsistent, "row", "rows")
                    issues_found = True
                    lines.append(
                        styler.error_text(
                            f"ID '{col}' has {inconsistent:,} inconsistent {row_label}!"
                        )
                    )
                    lines.append(
                        "      -> Same ID maps to different values; check ETL logic."
                    )

            if not issues_found:
                lines.append(f"   {styler.success_text('ID check passed')}")

        return "\n".join(lines)
