import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(n_max, n_slider, np):
    n = n_slider.value

    # Roll two 6-sided dice n times
    np.random.seed(0)
    die1 = np.random.randint(1, 7, size=n_max)
    die2 = np.random.randint(1, 7, size=n_max)
    sums = die1 + die2
    sums = sums[0:n]
    print(sums)
    return die1, die2, n, sums


@app.cell
def _(mo):
    n_max = 1000
    n_slider = mo.ui.number(
        start=1, stop=n_max, step=1, value=10, label="Number of 2d6 rolls"
    )

    mo.md(
        "## 🎲 2d6 Dice Roll Simulator\nMove the slider to change how many rolls are simulated:"
    )
    return n_max, n_slider


@app.cell
def _(mo, n, n_slider, np, plt, sums):
    fig, ax = plt.subplots(figsize=(8, 5))
    # MAX_COUNT = 5
    # Possible sums: 2 through 12
    bins = np.arange(1.5, 13.5, 1)
    counts, _, bars = ax.hist(sums, bins=bins)
    # ax.set_ylim(0, MAX_COUNT)

    # Color each bar differently
    cmap = plt.get_cmap("tab10")
    for i, bar in enumerate(bars):
        bar.set_color(cmap(i % 10))

    ax.set_xticks(range(2, 13))
    ax.set_xlabel("Sum of Two Dice")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Histogram of {n} Rolls of 2d6")
    ax.grid(axis="y", alpha=0.3)
    mo.vstack([n_slider, fig])
    return ax, bar, bars, bins, cmap, counts, fig, i


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
