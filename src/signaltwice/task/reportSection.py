from typing import Any, Dict, List, Optional


class ReportStyler:
    _HEADER = '\033[95m'
    _BLUE = '\033[94m'
    _CYAN = '\033[96m'
    _GREEN = '\033[92m'
    _YELLOW = '\033[93m'
    _RED = '\033[91m'
    _BOLD = '\033[1m'
    _RESET = '\033[0m'
    REPORT_WIDTH = 62

    def __init__(self, enable_color: bool = True) -> None:
        self.enable_color = enable_color

    def _style(self, code: str) -> str:
        return code if self.enable_color else ""

    def header_color(self) -> str:
        return self._style(self._HEADER)

    def blue(self) -> str:
        return self._style(self._BLUE)

    def cyan(self) -> str:
        return self._style(self._CYAN)

    def green(self) -> str:
        return self._style(self._GREEN)

    def yellow(self) -> str:
        return self._style(self._YELLOW)

    def red(self) -> str:
        return self._style(self._RED)

    def bold(self) -> str:
        return self._style(self._BOLD)

    def reset(self) -> str:
        return self._style(self._RESET)

    def header(self, text: str) -> str:
        width = self.REPORT_WIDTH
        safe_text = text[:width]
        border = "═" * width
        content = f"{safe_text}".center(width)
        return (
            f"\n{self.cyan()}{self.bold()}╔{border}╗\n"
            f"║{content}║\n"
            f"╚{border}╝{self.reset()}"
        )

    def sub_header(self, text: str) -> str:
        return f"\n{self.bold()}>> {text}{self.reset()}"

    def key_value(
        self,
        key: str,
        value: Any,
        unit: str = "",
        pad: int = 25,
        color: str = "",
        icon: Optional[str] = None,
    ) -> str:
        val_str = f"{value}"
        if unit:
            val_str += f" {unit}"
        label = f"{key:<{pad}}"
        if icon:
            gutter = f"{icon} "
        else:
            gutter = "   "
        label = f"{gutter}{label}"
        return f"{label} : {color}{val_str}{self.reset()}"

    def warning_text(self, text: str) -> str:
        return f"{self.yellow()}⚠️  {text}{self.reset()}"

    def error_text(self, text: str) -> str:
        return f"{self.red()}❌ {text}{self.reset()}"

    def success_text(self, text: str) -> str:
        return f"{self.green()}✅ {text}{self.reset()}"

    def count_noun(self, count: int, singular: str, plural: str) -> str:
        return singular if count == 1 else plural

    def separator(self, text: str = "End of Report") -> List[str]:
        width = self.REPORT_WIDTH
        border_width = width + 2
        return [
            f"\n{self.cyan()}{'=' * border_width}{self.reset()}",
            f"{text.center(width)}",
            f"{self.cyan()}{'=' * border_width}{self.reset()}\n",
        ]


class ReportSectionHandler:
    _strategies = {}

    @classmethod
    def register_strategy(cls, name: str):
        def wrapper(strategy_cls):
            cls._strategies[name] = strategy_cls
            return strategy_cls

        return wrapper

    @classmethod
    def generate_full_report(
        cls,
        raw_data: dict,
        execution_order: Optional[List[str]] = None,
        enable_color: bool = True,
    ) -> str:
        """Execute ordered strategies, then append any newly registered ones."""
        report_parts = []
        styler = ReportStyler(enable_color=enable_color)
        order = [
            "dataOverview",
            "missingValueReport",
            "numericStats",
            "categoricalStats",
            "dataIntegrity",
        ]
        base_order = execution_order or order
        keys_to_run = [k for k in base_order if k in cls._strategies]
        extra_keys = [k for k in cls._strategies.keys() if k not in keys_to_run]
        keys_to_run.extend(extra_keys)

        for name in keys_to_run:
            handler = cls._strategies[name]()
            metrics = handler.calculate(raw_data)
            section_text = handler.render(metrics, styler)
            report_parts.append(section_text)

        report_parts.extend(styler.separator())
        return "\n".join(report_parts)

class SafeExtractor:
    @staticmethod
    def mapping(value) -> dict:
        if isinstance(value, dict):
            return value
        return {}

    @staticmethod
    def mapping_from(container, key: str, default=None) -> dict:
        base = SafeExtractor.mapping(container)
        if default is None:
            default = {}
        return SafeExtractor.mapping(base.get(key, default))

    @staticmethod
    def deep_get(container, path: List[Any], default=None):
        current = container
        for key in path:
            if not isinstance(current, dict):
                return default
            current = current.get(key, default)
            if current is default:
                return default
        return current

    @staticmethod
    def deep_number(container, path: List[Any], default=0):
        return SafeExtractor.number(
            SafeExtractor.deep_get(container, path, default=default),
            default,
        )

    @staticmethod
    def deep_float(container, path: List[Any], default=0.0):
        value = SafeExtractor.deep_get(container, path, default=default)
        if value is None and default is None:
            return None
        return SafeExtractor.float(value, default)

    @staticmethod
    def deep_list(container, path: List[Any], default=None):
        if default is None:
            default = []
        return SafeExtractor.list(
            SafeExtractor.deep_get(container, path, default=default)
        )

    @staticmethod
    def truncate(text: Optional[str], max_len: int) -> str:
        if text is None:
            return ""
        if max_len <= 0:
            return ""
        if len(text) <= max_len:
            return text
        if max_len <= 3:
            return text[:max_len]
        return f"{text[:max_len - 3].rstrip()}..."

    @staticmethod
    def list(value) -> list:
        if isinstance(value, list):
            return value
        return []

    @staticmethod
    def number(value, default=0):
        if isinstance(value, (int, float)):
            return value
        return default

    @staticmethod
    def float(value, default=0.0):
        if isinstance(value, (int, float)):
            return float(value)
        return float(default)

    @staticmethod
    def describe_skewness(value):
        if not isinstance(value, (int, float)):
            return None
        if value > 1:
            return "right-skewed"
        if value < -1:
            return "left-skewed"
        return "balanced"


@ReportSectionHandler.register_strategy("dataOverview")
class DataOverviewStrategy:
    def calculate(self, raw_data: dict) -> dict:
        dtypes = SafeExtractor.mapping_from(raw_data, "get_dtypes")

        rows = SafeExtractor.deep_number(raw_data, ["get_shape", "rows"], 0)
        columns = SafeExtractor.deep_number(
            raw_data, ["get_shape", "columns"], 0
        )
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
        lines.append(
            styler.key_value(f"Total {row_label}", f"{rows:,}", icon="📊")
        )
        lines.append(
            styler.key_value(
                f"Total {column_label}", f"{columns:,}", icon="🧩"
            )
        )

        dupe_color = styler.red() if duplicate_rows > 0 else styler.green()
        dupe_value = f"{duplicate_rows:,}" if duplicate_rows > 0 else "0"
        lines.append(
            styler.key_value(
                duplicate_label, dupe_value, color=dupe_color, icon="👯"
            )
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


@ReportSectionHandler.register_strategy("missingValueReport")
class MissingValueStrategy:
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
        empty_rows_count = SafeExtractor.number(
            metrics.get("empty_rows_count"), 0
        )
        empty_rows_percent = SafeExtractor.float(
            metrics.get("empty_rows_percent"), 0.0
        )
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
            lines.append(
                f"   {styler.success_text('All columns are complete')}"
            )
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


@ReportSectionHandler.register_strategy("numericStats")
class NumericStatsStrategy:
    def calculate(self, raw_data: dict) -> dict:
        stats = SafeExtractor.mapping_from(raw_data, "get_descriptive_stats")
        dist = SafeExtractor.mapping_from(raw_data, "calc_distribution_shape")
        outliers = SafeExtractor.mapping_from(raw_data, "detect_outliers")

        columns = []
        for col, col_stats in stats.items():
            stat_map = SafeExtractor.mapping(col_stats)
            skew = SafeExtractor.deep_float(
                dist, [col, "skewness"], None
            )
            kurt = SafeExtractor.deep_float(
                dist, [col, "kurtosis"], None
            )

            columns.append(
                {
                    "column": col,
                    "mean": SafeExtractor.float(stat_map.get("mean"), 0.0),
                    "std": SafeExtractor.float(stat_map.get("std"), 0.0),
                    "min": stat_map.get("min"),
                    "max": stat_map.get("max"),
                    "skewness": skew,
                    "kurtosis": kurt,
                    "skewness_description": SafeExtractor.describe_skewness(
                        skew
                    ),
                    "outlier_count": SafeExtractor.deep_number(
                        outliers, [col, "outlier_count"], 0
                    ),
                }
            )

        return {
            "columns": columns,
        }

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
            outlier_count = SafeExtractor.number(
                item.get("outlier_count"), 0
            )

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
                f"{styler.yellow()}{outlier_count:,} (IQR)"
                f"{styler.reset()}"
                if outlier_count > 0
                else f"{styler.green()}0{styler.reset()}"
            )
            lines.append(f"   └─ 🚨 Outliers: {outlier_text}")

        return "\n".join(lines)


@ReportSectionHandler.register_strategy("categoricalStats")
class CategoricalStatsStrategy:
    def calculate(self, raw_data: dict) -> dict:
        cardinality = SafeExtractor.mapping_from(raw_data, "count_uniques")
        rare = SafeExtractor.mapping_from(raw_data, "check_rare_categories")
        noise = SafeExtractor.mapping_from(
            raw_data, "check_string_variants"
        )

        columns = []
        for col, count in cardinality.items():
            rare_count = SafeExtractor.deep_number(
                rare, [col, "rare_category_count"], 0
            )
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

        lines = [
            styler.header("4. Categorical & String Quality (CATEGORICAL)")
        ]

        if not columns:
            lines.append("  (No categorical/string columns detected)")
            return "\n".join(lines)

        for item in columns:
            col = item.get("column")
            display_col = SafeExtractor.truncate(str(col), 30)
            unique_values = SafeExtractor.number(
                item.get("unique_values"), 0
            )
            rare_count = SafeExtractor.number(
                item.get("rare_category_count"), 0
            )
            noise_count = SafeExtractor.number(
                item.get("variant_noise_count"), 0
            )

            lines.append(
                f"\n{styler.bold()}🔸 {display_col}{styler.reset()}"
            )
            lines.append(f"   ├─ 🏷️  Unique Values: {unique_values:,}")

            if rare_count == 0 and noise_count == 0:
                lines.append(
                    f"   └─ {styler.success_text('No rare categories or string noise')}"
                )
            else:
                issues = []
                if rare_count > 0:
                    issues.append(
                        styler.warning_text(
                            f"Rare categories: {rare_count}"
                        )
                    )
                else:
                    issues.append(
                        styler.success_text("Rare categories: None")
                    )

                if noise_count > 0:
                    issues.append(
                        styler.warning_text(
                            f"String noise: {noise_count}"
                        )
                    )
                else:
                    issues.append(styler.success_text("String noise: None"))

                lines.append(f"   └─ {' | '.join(issues)}")

        return "\n".join(lines)


@ReportSectionHandler.register_strategy("dataIntegrity")
class DataIntegrityStrategy:
    def calculate(self, raw_data: dict) -> dict:
        constants = SafeExtractor.mapping_from(
            raw_data, "check_constant_columns"
        )
        ids = SafeExtractor.mapping_from(raw_data, "check_id_integrity")

        constant_columns = [
            col
            for col, info in constants.items()
            if SafeExtractor.mapping(info).get("is_constant")
        ]

        id_checks = []
        for col, info in ids.items():
            inconsistent = SafeExtractor.deep_number(
                info, ["inconsistent_row_count"], 0
            )
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
        constant_columns = SafeExtractor.list(
            metrics.get("constant_columns")
        )
        id_checks = SafeExtractor.list(metrics.get("id_integrity"))

        lines = [styler.header("5. Data Integrity (DATA INTEGRITY)")]

        lines.append(styler.sub_header("Constant columns check"))
        if constant_columns:
            column_label = styler.count_noun(
                len(constant_columns), "column", "columns"
            )
            lines.append(
                styler.warning_text(
                    f"Found {len(constant_columns)} constant "
                    f"{column_label} (no variance):"
                )
            )
            lines.append(
                f"      {styler.red()}{', '.join(constant_columns)}"
                f"{styler.reset()}"
            )
            lines.append("      -> Consider removing them (no predictive value).")
        else:
            lines.append(
                f"   {styler.success_text('No constant columns found')}"
            )

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
                            f"ID '{col}' has {inconsistent:,} inconsistent "
                            f"{row_label}!"
                        )
                    )
                    lines.append(
                        "      -> Same ID maps to different values; check ETL logic."
                    )

            if not issues_found:
                lines.append(
                    f"   {styler.success_text('ID check passed')}"
                )

        return "\n".join(lines)
