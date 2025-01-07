import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import re
    import datetime
    from huggingface_hub import snapshot_download
    from pathlib import Path
    return Path, datetime, mo, pl, re, snapshot_download


@app.cell
def __(snapshot_download):
    repo = snapshot_download(
        repo_id='lmarena-ai/chatbot-arena-leaderboard',
        repo_type='space',
        allow_patterns='*.csv',
        cache_dir='.',
    )
    return (repo,)


@app.cell
def __(Path, pl, re, repo):
    csv_schema = {
        'key': pl.Utf8,
        'Model': pl.Utf8,
        'MT-bench (win rate %)': pl.Utf8,
        'MT-bench (score)': pl.Float32,
        'Arena Elo rating': pl.Float32,
        'MMLU': pl.Float32,
        'License': pl.Utf8,
        'Organization': pl.Utf8,
        'Link': pl.Utf8,
        'Knowledge cutoff date': pl.Utf8,
    }
    def scan_csv(csv_path):
        match = re.search(r'leaderboard_table_(\d{4}(0[1-9]|1[0-2])(0[1-9]|[12][0-9]|3[01])).csv', csv_path.name)
        datestamp = match.group(1)
        df = pl.scan_csv(
            csv_path,
            null_values='-',
            schema_overrides=csv_schema,
            ignore_errors=False
        )
        df = df.with_columns(pl.lit(datestamp).alias('date'))
        df = df.with_columns(
            pl.col('date')
            .str
            .to_date(format='%Y%m%d')
            .alias('date')
        )
        if 'Knowledge cutoff date' in df.collect_schema().names():
            df = df.with_columns(
                pl.col('Knowledge cutoff date')
                .str
                .to_date(format='%Y/%m', strict=False)
                .alias('Knowledge cutoff date')
            )
        return df

    csvs = Path(repo).glob('leaderboard_table_*.csv')

    dfs = [scan_csv(csv) for csv in csvs]
    return csv_schema, csvs, dfs, scan_csv


@app.cell
def __(dfs, pl):
    df = pl.concat(dfs, how='diagonal')
    df.collect()
    return (df,)


@app.cell
def __(datetime, df):
    import altair as alt

    data = df.collect().to_pandas()
    data['Organization'] = data['Organization'].fillna('Unknown')

    chart = (
        alt.Chart(data)
        .mark_line(
            color='Organization:N',
        )
        .encode(
            x=alt.X(
                'date:T',
                scale=alt.Scale(
                    domain=(
                        data['date'].min(),
                        datetime.datetime.today(),
                    )
                )
            ),
            y=alt.Y(
                'MMLU:Q',
                scale=alt.Scale(
                    domain=(
                        data['MMLU'].min(),
                        data['MMLU'].max()
                    )
                )
            ),
            shape='Organization:N',
            color='Model:N',
            tooltip=['Model', 'MMLU', 'Organization', 'Link']
        )
        .interactive()
    )
    chart = chart.properties(width='container', height=1000)
    chart
    return alt, chart, data


if __name__ == "__main__":
    app.run()
