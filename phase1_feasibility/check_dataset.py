"""Phase 1 feasibility check of the Financial PhraseBank training data.

Loads data/raw/all-data.csv and the four Sentences_*Agree.txt files, reports
descriptive statistics and writes them into feasibility_report.md. No cleaning,
splitting or modelling is done; the raw files are only read.
"""
import platform
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REPORT = Path(__file__).resolve().parent / "feasibility_report.md"
TXT_FILES = ["Sentences_50Agree.txt", "Sentences_66Agree.txt",
             "Sentences_75Agree.txt", "Sentences_AllAgree.txt"]


def md_table(df: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table (no extra dependency)."""
    cells = [[str(v).replace("|", "\\|") for v in row] for row in df.itertuples(index=False)]
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in cells])


def write_section(key: str, body: str) -> None:
    """Insert or replace the section delimited by markers for `key` in the report."""
    start, end = f"<!-- BEGIN {key} -->", f"<!-- END {key} -->"
    old = REPORT.read_text(encoding="utf-8") if REPORT.exists() else "# Phase 1 Feasibility Report\n"
    block = f"{start}\n{body.strip()}\n{end}"
    pattern = re.compile(re.escape(start) + ".*?" + re.escape(end), re.S)
    new = pattern.sub(lambda _: block, old) if pattern.search(old) else old.rstrip() + "\n\n" + block + "\n"
    REPORT.write_text(new, encoding="utf-8")


def load_csv(path: Path) -> tuple[pd.DataFrame, str, bool]:
    """Load the CSV trying UTF-8 then Latin-1; detect a header row heuristically."""
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, header=None, encoding=enc, names=["label", "text"])
            break
        except UnicodeDecodeError:
            continue
    # A header row would have a first value that never reappears as a label.
    has_header = df["label"].iloc[0] not in set(df["label"].iloc[1:])
    return (df.iloc[1:].reset_index(drop=True) if has_header else df), enc, has_header


def load_txt(path: Path) -> pd.DataFrame:
    """Parse a 'sentence@label' file, splitting on the last '@' and skipping empty lines."""
    lines = [ln.strip() for ln in path.read_text(encoding="latin-1").splitlines() if ln.strip()]
    return pd.DataFrame([ln.rsplit("@", 1) for ln in lines], columns=["text", "label"])


def label_summary(labels: pd.Series) -> str:
    """Format label counts with percentages, e.g. 'neutral 10 (50.0 %)'."""
    vc = labels.value_counts()
    return "; ".join(f"{k} {v} ({v / len(labels):.1%})" for k, v in vc.items())


def main() -> None:
    """Run all dataset checks and write the report section."""
    out = [f"## 1. Dataset check\n\nRun: {datetime.now():%Y-%m-%d %H:%M:%S} | "
           f"Python {platform.python_version()} | pandas {pd.__version__}\n"]
    out.append("Files in data/raw/: " + ", ".join(
        f"{p.name} ({p.stat().st_size:,} B)" for p in sorted(RAW.iterdir())) + "\n")

    df, enc, has_header = load_csv(RAW / "all-data.csv")
    out.append(f"### all-data.csv\n\n- Encoding that worked: **{enc}** (UTF-8 tried first)\n"
               f"- Header row detected: **{has_header}**; columns named `label`, `text`\n"
               f"- Rows: **{len(df)}**\n- Missing values: label {df['label'].isna().sum()}, "
               f"text {df['text'].isna().sum()}\n- Duplicate rows (label+text): "
               f"{df.duplicated().sum()}\n")
    vc = df["label"].value_counts()
    out.append(md_table(pd.DataFrame({"label": vc.index, "count": vc.values,
                                      "percent": [f"{v / len(df):.1%}" for v in vc.values]})))
    n_labels = df["label"].nunique()
    out.append(f"\nDistinct labels: {n_labels} -> at least three labels: **{n_labels >= 3}**\n")

    words = df["text"].str.split().str.len()
    out.append(f"Text length in words: min {words.min()}, median {words.median():.0f}, "
               f"mean {words.mean():.1f}, max {words.max()}\n")

    # Duplicated texts and whether the copies carry different labels.
    dup = df[df["text"].duplicated(keep=False)]
    groups = dup.groupby("text")["label"].agg(lambda s: sorted(set(s)))
    counts = dup.groupby("text").size()
    conflicts = int((groups.str.len() > 1).sum())
    out.append(f"#### Duplicated texts\n\nDistinct texts occurring more than once: {len(groups)} "
               f"({len(dup)} rows); with conflicting labels: **{conflicts}**\n")
    out.append(md_table(pd.DataFrame({
        "text (first 90 chars)": [t[:90] for t in groups.index],
        "occurrences": counts[groups.index].values,
        "labels": [", ".join(g) for g in groups],
        "conflict": ["**YES**" if len(g) > 1 else "no" for g in groups]})))

    out.append("\n#### Examples (2 per label)\n")
    ex = df.groupby("label").head(2).sort_values("label")
    out.append(md_table(ex.assign(text=ex["text"].str[:150])))

    # The four agreement-level .txt files.
    txt = {name: load_txt(RAW / name) for name in TXT_FILES}
    labels = sorted(df["label"].unique())
    rows = [{"file": name, "sentences": len(t)} | {lab: f"{(t['label'] == lab).sum()} "
            f"({(t['label'] == lab).mean():.1%})" for lab in labels} for name, t in txt.items()]
    out.append("\n### Agreement-level .txt files (Latin-1, split on last '@')\n")
    out.append(md_table(pd.DataFrame(rows)))

    # Compare all-data.csv to Sentences_50Agree.txt on stripped label + text.
    a = df.assign(text=df["text"].str.strip(), label=df["label"].str.strip())
    b = txt["Sentences_50Agree.txt"].apply(lambda c: c.str.strip())
    pairs_b = set(zip(b["label"], b["text"]))
    match = [p in pairs_b for p in zip(a["label"], a["text"])]
    mism = a[[not m for m in match]]
    # Re-test mismatches with every non-ASCII character mapped to '_'.
    def norm(s: str) -> str:
        return re.sub(r"[^\x00-\x7f]", "_", s)
    norm_b = {(lab, norm(t)) for lab, t in pairs_b}
    norm_ok = sum((lab, norm(t)) in norm_b for lab, t in zip(mism["label"], mism["text"]))
    same_labels = (a["label"] == b["label"]).all() if len(a) == len(b) else "n/a"
    out.append(f"\n### all-data.csv vs Sentences_50Agree.txt\n\n- Rows matching on label+text: "
               f"**{sum(match)} / {len(a)}**\n- Mismatching rows: {len(mism)}; of these, match "
               f"after mapping non-ASCII characters to '_': {norm_ok}\n- Labels identical row by "
               f"row (same order, same length): {same_labels}\n")
    ex_rows = []
    for i in mism.index[:2]:
        x, y = a.at[i, "text"], b.at[i, "text"]
        d = next((k for k, (c1, c2) in enumerate(zip(x, y)) if c1 != c2), 0)
        ex_rows.append({"row": i, "label csv/txt": f"{a.at[i, 'label']}/{b.at[i, 'label']}",
                        "csv snippet": ascii(x[max(d - 25, 0):d + 15]),
                        "txt snippet": ascii(y[max(d - 25, 0):d + 15])})
    if ex_rows:
        out.append(md_table(pd.DataFrame(ex_rows)))

    out.append("\n### Observations\n")
    out.append(f"- all-data.csv decodes as {enc}, has {'a' if has_header else 'no'} header row, "
               f"{len(df)} rows and {n_labels} labels ({label_summary(df['label'])}).")
    out.append(f"- {df.duplicated().sum()} exact duplicate rows; {len(groups)} texts occur more "
               f"than once, {conflicts} of them with conflicting labels.")
    out.append(f"- {sum(match)} of {len(a)} rows match Sentences_50Agree.txt on label+text; "
               f"{norm_ok} of the {len(mism)} mismatches match once non-ASCII characters "
               f"are mapped to '_'; labels identical row by row: {same_labels}.")
    out.append("- Sentence counts per agreement file: "
               + ", ".join(f"{k} {len(v)}" for k, v in txt.items()) + ".")
    write_section("dataset", "\n".join(out))
    print(f"Dataset section written to {REPORT}")


if __name__ == "__main__":
    main()
