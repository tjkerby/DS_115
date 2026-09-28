"""
Generate synthetic DS 115 Day 2 student-success data, summary tables, figures,
and a simple predictive-model demo.

Run from the course repository root:
    python slides/day2_assets/day2_student_success_data.py

Outputs (all under slides/day2_assets/):
    data/students_clean.csv
    data/students_messy.csv
    outputs/*.csv
    figures/*.png
    snippets/*.md
"""

from __future__ import annotations

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "This script needs scikit-learn. Install it with: pip install scikit-learn"
    ) from exc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUT_DIR = ROOT / "outputs"
SNIPPET_DIR = ROOT / "snippets"
for directory in (DATA_DIR, FIG_DIR, OUT_DIR, SNIPPET_DIR):
    directory.mkdir(exist_ok=True)

RNG = np.random.default_rng(115)
N = 750

YEARS = ["First-year", "Sophomore", "Junior", "Senior"]
MAJORS = ["Business", "Computer Science", "Psychology", "Biology", "Statistics", "English", "Exercise Science", "Undeclared"]
HOUSING = ["On campus", "Off campus", "With family"]


def clamp(values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(values, lo), hi)


def make_clean_data() -> pd.DataFrame:
    year = RNG.choice(YEARS, N, p=[0.28, 0.25, 0.24, 0.23])
    major = RNG.choice(MAJORS, N, p=[0.16, 0.15, 0.12, 0.13, 0.10, 0.10, 0.12, 0.12])
    housing = RNG.choice(HOUSING, N, p=[0.42, 0.43, 0.15])

    # Year-dependent shifts so aggregations by year are visibly different
    # rather than identical apart from sampling noise.
    year_idx = pd.Series(year).map({y: i for i, y in enumerate(YEARS)}).to_numpy()
    sleep_shift = np.array([-0.25, -0.05, 0.10, 0.20])[year_idx]
    study_shift = np.array([1.0, 0.0, -0.5, -1.0])[year_idx]
    work_shift = np.array([-3.0, -1.0, 1.5, 3.5])[year_idx]
    attend_shift = np.array([0.04, 0.02, -0.02, -0.05])[year_idx]
    credit_shift = np.array([0.5, 0.0, -0.5, -1.0])[year_idx]
    gpa_shift = np.array([-0.08, 0.02, 0.08, 0.12])[year_idx]

    previous_gpa = clamp(RNG.normal(3.18, 0.42, N) + gpa_shift, 1.6, 4.0)
    study_hours = clamp(RNG.normal(12.5, 4.8, N) + study_shift, 1, 32).round(1)
    sleep_hours = clamp(RNG.normal(6.75, 0.85, N) + sleep_shift, 4.0, 9.5).round(1)
    work_hours = clamp(RNG.gamma(2.0, 5.5, N) + work_shift, 0, 34).round(1)
    attendance_rate = clamp(RNG.normal(0.88, 0.08, N) + attend_shift, 0.55, 1.0).round(2)
    commute_minutes = np.where(
        housing == "On campus",
        RNG.normal(8, 4, N),
        np.where(housing == "With family", RNG.normal(24, 12, N), RNG.normal(17, 9, N)),
    )
    commute_minutes = clamp(commute_minutes, 0, 60).round(0).astype(int)
    credits = clamp(
        RNG.choice([12, 13, 14, 15, 16, 17, 18], N, p=[0.10, 0.11, 0.17, 0.23, 0.20, 0.13, 0.06]).astype(float)
        + credit_shift,
        12, 18,
    ).round(0).astype(int)
    phone_hours = clamp(RNG.normal(4.2, 1.5, N), 0.5, 9.0).round(1)

    year_effect = pd.Series(year).map({"First-year": -0.06, "Sophomore": 0.00, "Junior": 0.04, "Senior": 0.08}).to_numpy()
    sleep_bonus = -0.045 * (sleep_hours - 7.2) ** 2 + 0.055
    work_penalty = -0.006 * work_hours
    phone_penalty = -0.025 * np.maximum(phone_hours - 4.5, 0)
    attendance_bonus = 0.95 * (attendance_rate - 0.85)
    study_bonus = 0.018 * study_hours
    credit_penalty = -0.015 * np.maximum(credits - 16, 0)
    noise = RNG.normal(0, 0.22, N)

    current_gpa = (
        0.55 * previous_gpa
        + study_bonus
        + attendance_bonus
        + sleep_bonus
        + work_penalty
        + phone_penalty
        + credit_penalty
        + year_effect
        + 0.90
        + noise
    )
    current_gpa = clamp(current_gpa, 0.0, 4.0).round(2)

    df = pd.DataFrame(
        {
            "student_id": np.arange(10001, 10001 + N),
            "year": year,
            "major": major,
            "housing": housing,
            "previous_gpa": previous_gpa.round(2),
            "current_gpa": current_gpa,
            "study_hours_per_week": study_hours,
            "sleep_hours_per_night": sleep_hours,
            "work_hours_per_week": work_hours,
            "attendance_rate": attendance_rate,
            "commute_minutes": commute_minutes,
            "credits": credits,
            "phone_hours_per_day": phone_hours,
        }
    )
    return df


def make_messy_data(clean: pd.DataFrame) -> pd.DataFrame:
    messy = clean.sample(24, random_state=115).copy().reset_index(drop=True)
    for col in ["study_hours_per_week", "work_hours_per_week"]:
        messy[col] = messy[col].astype(object)
    messy.loc[1, "sleep_hours_per_night"] = np.nan
    messy.loc[2, "study_hours_per_week"] = "twelve"
    messy.loc[3, "sleep_hours_per_night"] = -3
    messy.loc[4, "current_gpa"] = 7.8
    messy.loc[5, "attendance_rate"] = 95
    messy.loc[6, "year"] = "jr"
    messy.loc[7, "year"] = "JUNIOR"
    messy.loc[8, "major"] = "bio"
    messy.loc[9, "work_hours_per_week"] = ""
    messy.loc[10, "housing"] = "off-campus"
    messy.loc[11, "student_id"] = messy.loc[10, "student_id"]
    return messy


def add_sleep_group(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sleep_group"] = np.where(out["sleep_hours_per_night"] >= 7, "7+ hours", "< 7 hours")
    return out


def save_tables(df: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "question": [
                "How many students are in the dataset?",
                "What is the average GPA?",
                "What is the median GPA?",
                "What proportion attend at least 90% of class meetings?",
                "What proportion work more than 20 hours/week?",
                "What is the average sleep per night?",
            ],
            "answer": [
                len(df),
                round(df["current_gpa"].mean(), 2),
                round(df["current_gpa"].median(), 2),
                round((df["attendance_rate"] >= 0.90).mean(), 2),
                round((df["work_hours_per_week"] > 20).mean(), 2),
                round(df["sleep_hours_per_night"].mean(), 2),
            ],
        }
    )
    summary.to_csv(OUT_DIR / "summary_answers.csv", index=False)

    by_year = (
        df.groupby("year", observed=True)
        .agg(
            students=("student_id", "count"),
            avg_sleep=("sleep_hours_per_night", "mean"),
            avg_study=("study_hours_per_week", "mean"),
            avg_gpa=("current_gpa", "mean"),
        )
        .reindex(YEARS)
        .round(2)
        .reset_index()
    )
    by_year.to_csv(OUT_DIR / "aggregation_by_year.csv", index=False)

    by_sleep = (
        add_sleep_group(df)
        .groupby("sleep_group", observed=True)
        .agg(students=("student_id", "count"), avg_gpa=("current_gpa", "mean"), median_gpa=("current_gpa", "median"))
        .round(2)
        .reset_index()
    )
    by_sleep.to_csv(OUT_DIR / "gpa_by_sleep_group.csv", index=False)


def format_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=16, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def save_figures(df: pd.DataFrame, predictions: pd.DataFrame, model_metrics: dict[str, float]) -> None:
    by_year = pd.read_csv(OUT_DIR / "aggregation_by_year.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    year_bars = ax.bar(by_year["year"], by_year["avg_gpa"])
    format_axes(ax, "Average GPA by class year", ylabel="Average current GPA")
    ax.set_ylim(0, 3)
    ax.bar_label(year_bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "average_gpa_by_year.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.scatter(df["study_hours_per_week"], df["current_gpa"], alpha=0.45, s=18)
    format_axes(ax, "Study time and GPA", xlabel="Study hours per week", ylabel="Current GPA")
    ax.set_ylim(1.7, 4.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "study_vs_gpa.png")
    plt.close(fig)

    sleep = pd.read_csv(OUT_DIR / "gpa_by_sleep_group.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    sleep_bars = ax.bar(sleep["sleep_group"], sleep["avg_gpa"])
    format_axes(ax, "Summarizing many rows into two groups", ylabel="Average current GPA")
    ax.set_ylim(0, 3)
    ax.bar_label(sleep_bars, fmt="%.2f", padding=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gpa_by_sleep_group.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(predictions["actual_gpa"], predictions["predicted_gpa"], alpha=0.55, s=20)
    ax.plot([2.0, 4.0], [2.0, 4.0], linestyle="--")
    format_axes(
        ax,
        f"Prediction is useful, but imperfect\nMAE = {model_metrics['mae']:.2f}, R² = {model_metrics['r2']:.2f}",
        xlabel="Actual GPA",
        ylabel="Predicted GPA",
    )
    ax.set_xlim(2.0, 4.05)
    ax.set_ylim(2.0, 4.05)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "actual_vs_predicted_gpa.png")
    plt.close(fig)


def fit_model(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    features = [
        "previous_gpa",
        "study_hours_per_week",
        "sleep_hours_per_night",
        "work_hours_per_week",
        "attendance_rate",
        "credits",
        "phone_hours_per_day",
        "year",
        "housing",
    ]
    target = "current_gpa"
    X = df[features]
    y = df[target]

    numeric_features = [
        "previous_gpa",
        "study_hours_per_week",
        "sleep_hours_per_night",
        "work_hours_per_week",
        "attendance_rate",
        "credits",
        "phone_hours_per_day",
    ]
    categorical_features = ["year", "housing"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=115)
    model.fit(X_train, y_train)
    pred = np.clip(model.predict(X_test), 0, 4)

    pred_df = X_test.copy()
    pred_df["actual_gpa"] = y_test.to_numpy()
    pred_df["predicted_gpa"] = pred.round(2)
    pred_df["absolute_error"] = np.abs(pred_df["actual_gpa"] - pred_df["predicted_gpa"]).round(2)
    pred_df = pred_df.sort_values("absolute_error")
    pred_df.to_csv(OUT_DIR / "model_predictions.csv", index=False)

    metrics = {
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }

    maya = pd.DataFrame(
        [
            {
                "previous_gpa": 3.45,
                "study_hours_per_week": 15.0,
                "sleep_hours_per_night": 7.2,
                "work_hours_per_week": 8.0,
                "attendance_rate": 0.95,
                "credits": 15,
                "phone_hours_per_day": 3.8,
                "year": "Sophomore",
                "housing": "Off campus",
            }
        ]
    )
    maya_prediction = float(np.clip(model.predict(maya)[0], 0, 4))
    maya_out = maya.copy()
    maya_out["predicted_gpa"] = round(maya_prediction, 2)
    maya_out.to_csv(OUT_DIR / "maya_prediction.csv", index=False)

    return pred_df, maya_out, metrics



def save_markdown_snippets(df: pd.DataFrame, messy: pd.DataFrame, maya: pd.DataFrame) -> None:
    """Write static Markdown fragments consumed by the Quarto slides."""
    cols = [
        "student_id", "year", "previous_gpa", "current_gpa",
        "study_hours_per_week", "sleep_hours_per_night",
        "work_hours_per_week", "attendance_rate",
    ]
    messy_cols = [
        "student_id", "year", "major", "current_gpa",
        "study_hours_per_week", "sleep_hours_per_night",
        "work_hours_per_week", "attendance_rate",
    ]

    def write_table(name: str, frame: pd.DataFrame) -> None:
        (SNIPPET_DIR / name).write_text(frame.to_markdown(index=False) + "\n")

    write_table("student_preview.md", df[cols].head(8))
    write_table("one_student.md", df[cols].iloc[[0]])
    write_table("summary_answers.md", pd.read_csv(OUT_DIR / "summary_answers.csv"))
    write_table("aggregation_by_year.md", pd.read_csv(OUT_DIR / "aggregation_by_year.csv"))
    write_table("gpa_by_sleep_group.md", pd.read_csv(OUT_DIR / "gpa_by_sleep_group.csv"))

    maya_profile = maya.drop(columns=["predicted_gpa"]).T.reset_index()
    maya_profile.columns = ["variable", "value"]
    write_table("maya_profile.md", maya_profile)
    write_table("messy_preview.md", messy[messy_cols].head(12))

    avg_gpa = float(df["current_gpa"].mean())
    claim_text = (
        f"Our 750 observed students average **{avg_gpa:.2f} GPA**.\n\n"
        "::: {.incremental}\n"
        f"- These 750 students averaged **{avg_gpa:.2f} GPA**.\n"
        f"- BYU students average **{avg_gpa:.2f} GPA**.\n"
        f"- College students average **{avg_gpa:.2f} GPA**.\n"
        ":::\n"
    )
    (SNIPPET_DIR / "sample_claims.md").write_text(claim_text)

    pred = float(maya.loc[0, "predicted_gpa"])
    prediction_text = (
        "::: {.giant .center}\n"
        f"{pred:.2f} GPA\n"
        ":::\n\n"
        "::: {.supporting .center}\n"
        "A useful estimate, not a certainty.\n"
        ":::\n"
    )
    (SNIPPET_DIR / "maya_prediction.md").write_text(prediction_text)

def main() -> None:
    clean = make_clean_data()
    messy = make_messy_data(clean)

    clean.to_csv(DATA_DIR / "students_clean.csv", index=False)
    messy.to_csv(DATA_DIR / "students_messy.csv", index=False)

    save_tables(clean)
    predictions, maya, metrics = fit_model(clean)
    save_figures(clean, predictions, metrics)
    save_markdown_snippets(clean, messy, maya)

    print("Created synthetic data, figures, and slide snippets:")
    print(f"  {DATA_DIR / 'students_clean.csv'}")
    print(f"  {DATA_DIR / 'students_messy.csv'}")
    print(f"  {OUT_DIR / 'summary_answers.csv'}")
    print(f"  {OUT_DIR / 'aggregation_by_year.csv'}")
    print(f"  {OUT_DIR / 'gpa_by_sleep_group.csv'}")
    print(f"  {OUT_DIR / 'model_predictions.csv'}")
    print(f"  {OUT_DIR / 'maya_prediction.csv'}")
    print(f"  figures in {FIG_DIR}")
    print(f"Maya prediction: {maya.loc[0, 'predicted_gpa']:.2f}")
    print(f"Model MAE: {metrics['mae']:.2f}; R2: {metrics['r2']:.2f}")


if __name__ == "__main__":
    main()
