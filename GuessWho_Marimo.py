import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Setup all the buttons/sliders/etc

    import matplotlib.pyplot as plt
    import random
    import numpy as np
    import math
    import marimo as mo

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from mpl_toolkits.axes_grid1 import (
        make_axes_locatable,
    )  # for colorbar alignment
    import matplotlib.patheffects as path_effects


    MAX_CHARS = 31
    DEFAULT_CHARS = 20

    # sliders to choose the starting number of characters
    P1_max_chars_slider = mo.ui.number(
        start=2,
        stop=MAX_CHARS,
        value=DEFAULT_CHARS,
        label="Player 1 Starting Characters:",
    )
    P2_max_chars_slider = mo.ui.number(
        start=2,
        stop=MAX_CHARS,
        value=DEFAULT_CHARS,
        label="Player 2 Starting Characters:",
    )


    # buttons to restart the game and confirm guess
    restart_button = mo.ui.run_button(label="Restart Game")
    guess_button = mo.ui.run_button(label="Make this Guess!", full_width=True)

    # marimo state variables to track the secrete character for each player
    P1_get_secret_char, P1_set_secret_char = mo.state(0)
    P2_get_secret_char, P2_set_secret_char = mo.state(0)
    random_seed_getter, random_seed_setter = mo.state(0)

    # track the list of guesses the player has made and who has been eliminated already by each player
    P1_guess_list = []
    P2_guess_list = []

    P1_eliminated = set()
    P2_eliminated = set()

    # P1_set_secret_char(random.randint(1, DEFAULT_CHARS))
    # P2_set_secret_char(random.randint(1, DEFAULT_CHARS))
    return (
        BoundaryNorm,
        DEFAULT_CHARS,
        ListedColormap,
        MAX_CHARS,
        P1_eliminated,
        P1_get_secret_char,
        P1_guess_list,
        P1_max_chars_slider,
        P1_set_secret_char,
        P2_eliminated,
        P2_get_secret_char,
        P2_guess_list,
        P2_max_chars_slider,
        P2_set_secret_char,
        Rectangle,
        guess_button,
        make_axes_locatable,
        math,
        mo,
        np,
        path_effects,
        plt,
        random,
        random_seed_getter,
        random_seed_setter,
        restart_button,
    )


@app.cell
def _(P1_max_chars_slider, mo):
    # range sldier to allow guesses
    range_slider = mo.ui.range_slider(
        start=1, stop=P1_max_chars_slider.value, full_width=True
    )
    return (range_slider,)


@app.cell
def _():
    return


@app.cell
def _(
    BoundaryNorm,
    History,
    ListedColormap,
    P1_max_chars_slider,
    P2_max_chars_slider,
    Rectangle,
    b_star,
    dropdown_options,
    history_checkbox,
    make_axes_locatable,
    mo,
    np,
    p_star,
    path_effects,
    play_by_play,
    plot_dropdown,
    plt,
    random,
    region_labels_checkbox,
    regions_checkbox,
):
    # Example 20×20 matrix with values 1–16
    N = 31

    fig, ax = plt.subplots(figsize=(6, 6))

    if plot_dropdown.value == dropdown_options[0]:
        matrix = b_star[1 : N + 1, 1 : N + 1].T  # assuming p_star is defined

        # Define colormap and normalization
        random.seed(3)

        colors = [
            "#215E61",
            "#3F969B",
            "#8CF5F9",
            "#F9F54C",
            "#FFEBD2",
            "#DBDFEA",
            "#ACB2D6",
            "#8294C4",
            "#2ED3E1",
            "#A459D0",
            "#F267AC",
            "#FFB84C",
            "#FDFEAE",
            "#C3ECC0",
            "#AAC7A7",
        ]

        random.shuffle(colors)

        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(1, 17) - 0.5, cmap.N)

        im = ax.imshow(
            np.round(matrix).astype(int),
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="none",
        )
        # --- Make colorbar same height as plot ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        # Create colorbar
        # cbar = fig.colorbar(im)

        # Place ticks at the center of each color block
        cbar.set_ticks(np.arange(1, 16))  # centers correspond to integer values
        cbar.set_ticklabels(np.arange(1, 16))
        cbar.set_label("Player 1's bid size")

        ax.set_title(
            "Optimal bid size for Player 1 (on Player 1's turn): $b^\star(n,m)$"
        )

    elif plot_dropdown.value == dropdown_options[1]:
        matrix = p_star[1 : N + 1, 1 : N + 1].T  # assuming p_star is defined

        # Define discrete boundaries for the color normalization (0 → 1 in steps of 0.05)
        values = np.arange(0.0, 1.01, 0.05)  # 0.0, 0.05, ..., 1.0

        # Use continuous inferno colormap
        cmap = plt.cm.rainbow

        # BoundaryNorm ensures that each interval maps properly
        norm = BoundaryNorm(values, ncolors=cmap.N, clip=False)

        # Display the matrix
        im = ax.imshow(
            matrix, cmap=cmap, norm=norm, origin="lower", interpolation="none"
        )
        # Make colorbar same height as plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)

        # Place ticks at discrete values
        cbar.set_ticks(values)
        cbar.set_ticklabels([f"{v:.2f}" for v in values])
        cbar.set_label("Player 1 win probability")

        ax.set_title(
            "Optimal win probability for Player 1 (on Player 1's turn) $p^\star(n,m)$"
        )

        # --- Add crosshatching for values near 50% ---
        threshold = 0.025
        for ii in range(N):
            for jj in range(N):
                if (
                    abs(matrix[jj, ii] - 0.5) <= threshold
                ):  # note: matrix is displayed with origin='lower'
                    ax.add_patch(
                        Rectangle(
                            (ii - 0.5, jj - 0.5),
                            1,
                            1,
                            facecolor="none",
                            edgecolor="black",
                            hatch="...",
                            linewidth=0.0,
                            zorder=6,  # above imshow but below region outlines
                        )
                    )

        # Height of the hatch band corresponds to ±0.02 around 0.5
        band_bottom = 0.5 - threshold
        band_top = 0.5 + threshold

        cbar.ax.add_patch(
            Rectangle(
                (0, band_bottom),
                1,
                band_top - band_bottom,
                facecolor="none",
                edgecolor="black",
                hatch="...",
                linewidth=1,
                transform=cbar.ax.transAxes,
                zorder=10,
            )
        )
    elif plot_dropdown.value == dropdown_options[2]:
        # --- Blank grid ---
        blank_matrix = np.zeros((N, N))  # all zeros, will appear blank
        ax.imshow(
            blank_matrix,
            cmap="Greys",
            origin="lower",
            interpolation="none",
            alpha=0,
        )  # invisible cells

    # Configure grid/ticks
    ax.set_xticks(np.arange(N))
    ax.set_yticks(np.arange(N))
    ax.set_xticklabels(np.arange(1, N + 1), fontsize=8)
    ax.set_yticklabels(np.arange(1, N + 1))

    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.add_patch(
        Rectangle(
            (0 - 0.5, 0 - 0.5),
            1,
            N,
            facecolor="none",
            edgecolor="black",
            hatch="**",
            linewidth=0.0,
            zorder=5,
        )
    )

    # Crosshatch the y=1 row (the first row of boxes)
    ax.add_patch(
        Rectangle(
            (0 - 0.5, 0 - 0.5),
            N,
            1,
            facecolor="none",
            edgecolor="black",
            hatch="**",
            linewidth=0.0,
            zorder=5,
        )
    )


    def draw_region(x1, y1, x2, y2, label):
        # Add rectangle outline
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        ax.add_patch(
            Rectangle(
                (x1 - 1.5, y1 - 1.5),  # bottom-left corner (x1, y1)
                x2 + 0.5 - x1 + 0.5,  # width
                y2 + 0.5 - y1 + 0.5,  # height
                fill=False,  # only outline
                edgecolor="black",
                linewidth=4.0,
                zorder=10,  # draw above other elements
            )
        )
        # Center label
        cx, cy = x1 - 1.5 + width / 2, y1 - 1.5 + height / 2
        if region_labels_checkbox.value:  # draw label
            txt = ax.text(
                cx,
                cy,
                label,
                fontsize=24,
                color="white",
                ha="center",
                va="center",
                zorder=11,
            )
            # Add black outline
            txt.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2.5, foreground="black"),
                    path_effects.Normal(),
                ]
            )


    # region: y=3 to y=4, x=4 to x=max_value (i.e., x=N)
    y1, y2 = 3, 4
    x1, x2 = 4, N + 1  # assuming N = max x value

    # regions = True
    if regions_checkbox.value:
        # draw horizontal regions
        draw_region(2, 2, N + 1, 2, "$W_0$")
        draw_region(4, 3, N + 1, 4, "$W_1$")
        draw_region(8, 5, N + 1, 8, "$W_2$")
        draw_region(16, 9, N + 1, 16, "$W_3$")

        # draw vertical regions
        draw_region(2, 3, 3, N + 1, "$U_0$")
        draw_region(4, 5, 7, N + 1, "$U_1$")
        draw_region(8, 9, 15, N + 1, "$U_2$")
        draw_region(16, 17, 31, N + 1, "$U_3$")
        # draw_region(2,3,3,N+1)

    # Draw dashed diagonal y = x
    ax.plot(
        [-0.5, N - 0.5],
        [-0.5, N - 0.5],
        color="black",
        linestyle="--",
        linewidth=1.5,
        zorder=8,
    )
    ax.set_xlim(0 - 0.5, N - 0.5)
    ax.set_ylim(0 - 0.5, N - 0.5)


    # --- Axis labels ---
    ax.set_xlabel(
        "Number of people remaining for Player 1: $n$", fontsize=12, labelpad=10
    )
    ax.set_ylabel(
        "Number of people remaining for Player 2: $m$", fontsize=12, labelpad=10
    )

    # plt.savefig("bstar_yes_regions.png")
    # plt.show()
    # print("plot ready")

    if history_checkbox.value:
        # --- Plot numbered circles for the Game_History ---
        for ii, (x, y) in enumerate(History, start=1):
            # Draw a white circle with black edge
            circ = plt.Circle(
                (x - 1, y - 1), 0.45, color="white", ec="black", lw=1.5, zorder=20
            )
            ax.add_patch(circ)

            # Add the step number at the center of the circle
            ax.text(
                x - 1,
                y - 1,
                str(ii - 1),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="black",
                zorder=21,
            )

        # --- Connect points with arrows + alternating labels ---
        p1_guess = 1
        p2_guess = 1

        for ii in range(len(History) - 1):
            (x1, y1) = History[ii]
            (x2, y2) = History[ii + 1]

            # Convert to plot coordinates
            x1p, y1p = x1 - 1, y1 - 1
            x2p, y2p = x2 - 1, y2 - 1

            # Choose label using parity of ii + compute guess number from ii
            if ii % 2 == 0:
                guess_num = ii // 2 + 1
                label = f"P1 #{guess_num}"
                text_y_offset = 0.35
                text_x_offset = 0
                my_linestyle = "-"
            else:
                guess_num = ii // 2 + 1
                label = f"P2 #{guess_num}"
                text_y_offset = 0
                text_x_offset = 0.35
                my_linestyle = ":"

            # Draw arrow
            if x1 - x2 >= 2 or y1 - y2 >= 2:
                ax.annotate(
                    "",
                    xy=(x2p, y2p),
                    xytext=(x1p, y1p),
                    arrowprops=dict(
                        arrowstyle="->",
                        linestyle=my_linestyle,
                        lw=2.2,
                        color="black",
                        shrinkA=5,
                        shrinkB=5,
                    ),
                    zorder=30,
                )

            # Label placement (midpoint)
            xm = (x1p + x2p) / 2
            ym = (y1p + y2p) / 2

        # ax.text(
        #     xm + text_x_offset,
        #     ym + text_y_offset,
        #     label,
        #     ha="center",
        #     va="center",
        #     fontsize=10,
        #     fontweight="bold",
        #     color="black",
        #     zorder=31,
        # )

    my_checkboxes = (
        [regions_checkbox, region_labels_checkbox]
        if regions_checkbox.value
        else [regions_checkbox]
    )

    if len(play_by_play) == 0:
        play_by_play.append("(Nothing has happened yet)")

    mo.accordion(
        {
            "Instructions": mo.md("""
    **Guess Who?** is a game where you ask Yes/No questions to find your opponent's secret character by process of elimination.

    In this **"Mathematician's Guess Who?"** your secret character is a number from 1 to 20, and you ask questions of the form **"Is your number between x and y?"**

    Use the slider below to choose your guess range, then press **"Make this Guess!"** to confirm.

    ### Instructions
    - Use the slider below to choose your guess range.
    - Press **"Make this Guess!"** to confirm your guess.
    - Use the **"Restart Game"** button to play again.
    - See the **Graphs** tab for plots from the optimal-strategy paper: https://arxiv.org/abs/1509.03327

    Good luck and have fun!
    """),
            "Change Starting Values": mo.vstack(
                [P1_max_chars_slider, P2_max_chars_slider]
            ),
            "Play-by-Play": mo.plain_text("\n".join(play_by_play)),
            "Graphs": mo.vstack(
                [
                    mo.hstack(
                        [
                            plot_dropdown,
                            mo.hstack(my_checkboxes, justify="start"),
                            history_checkbox,
                        ]
                    ),
                    mo.mpl.interactive(plt.gca()),
                ],
            ),
        }
    )
    return (
        N,
        ax,
        band_bottom,
        band_top,
        blank_matrix,
        cax,
        cbar,
        circ,
        cmap,
        colors,
        divider,
        draw_region,
        fig,
        guess_num,
        ii,
        im,
        jj,
        label,
        matrix,
        my_checkboxes,
        my_linestyle,
        norm,
        p1_guess,
        p2_guess,
        text_x_offset,
        text_y_offset,
        threshold,
        values,
        x,
        x1,
        x1p,
        x2,
        x2p,
        xm,
        y,
        y1,
        y1p,
        y2,
        y2p,
        ym,
    )


@app.cell
def _(
    P1_eliminated,
    P1_get_secret_char,
    P1_guess_list,
    P1_max_chars_slider,
    P1_set_secret_char,
    P2_eliminated,
    P2_get_secret_char,
    P2_guess_list,
    P2_max_chars_slider,
    P2_set_secret_char,
    b_star,
    guess_button,
    mo,
    random,
    random_seed_getter,
    random_seed_setter,
    range_slider,
    restart_button,
):
    # handle events when the button is pressed
    if guess_button.value:
        print(range_slider.value)
        P1_guess_list.append(range_slider.value)

    if restart_button.value:
        random.seed(random_seed_getter())
        P1_set_secret_char(random.randint(1, P1_max_chars_slider.value))
        P2_set_secret_char(random.randint(1, P2_max_chars_slider.value))
        P1_guess_list.clear()
        random_seed_setter(lambda value: value + 1)


    P2_guess_list.clear()
    P2_eliminated.clear()
    P1_eliminated.clear()
    History = [(P1_max_chars_slider.value, P2_max_chars_slider.value)]
    play_by_play = []
    computers_latest_guess = "(No guesses yet, Human goes first)"
    humans_latest_guess = "(No guesses yet)"

    if (
        guess_button.value
    ):  # include this to make sure the code runs whenever the button is pressed!
        pass

    P1_chars_remaining = P1_max_chars_slider.value - len(P1_eliminated)
    P2_chars_remaining = P2_max_chars_slider.value - len(P2_eliminated)
    for i in range(len(P1_guess_list)):
        humans_latest_guess = f"**Question:** Is your number between {P1_guess_list[i][0]} and {P1_guess_list[i][1]} inclusive?<br>(i.e. {P1_guess_list[i][0]} <= x <= {P1_guess_list[i][1]})?"
        play_by_play.append(
            f"Human Guess #{i + 1}: Is your number between {P1_guess_list[i][0]} and {P1_guess_list[i][1]} inclusive? (i.e. {P1_guess_list[i][0]} <= x <= {P1_guess_list[i][1]})?"
        )
        if P1_guess_list[i][0] <= P1_get_secret_char() <= P1_guess_list[i][1]:
            humans_latest_guess += "<br>**Answer:** YES!"
            play_by_play.append(f"  Answer: YES!")
            for j in range(1, P1_guess_list[i][0]):
                P1_eliminated.add(j)
            for j in range(P1_guess_list[i][1] + 1, P1_max_chars_slider.value + 1):
                P1_eliminated.add(j)
        else:
            humans_latest_guess += "<br>**Answer:** NO!"
            play_by_play.append(f"  Answer: NO!")
            for j in range(P1_guess_list[i][0], P1_guess_list[i][1] + 1):
                P1_eliminated.add(j)

        display_P1_eliminated = P1_eliminated if len(P1_eliminated) > 0 else "None"
        play_by_play.append(f"    Eliminated so far: {display_P1_eliminated}")

        P1_chars_remaining = P1_max_chars_slider.value - len(P1_eliminated)
        play_by_play.append(f"    Num options remaining: {P1_chars_remaining}")
        if P1_chars_remaining == 1:
            play_by_play.append("    CONGRATULATIONS, YOU WIN!")
            break

        P2_chars_remaining = P2_max_chars_slider.value - len(P2_eliminated)
        History.append((P1_chars_remaining, P2_chars_remaining))

        CPU_guess_size = b_star[
            P2_chars_remaining, P1_chars_remaining
        ]  # int(P2_chars_remaining // 2)
        smallest_non_eliminated = min(
            set(range(1, P2_max_chars_slider.value)) - P2_eliminated
        )

        P2_guess_list.append(
            (smallest_non_eliminated, smallest_non_eliminated + CPU_guess_size - 1)
        )

        computers_latest_guess = f"**Question:** Is your number between {P2_guess_list[i][0]} and {P2_guess_list[i][1]} inclusive? (i.e. {P2_guess_list[i][0]} <= x <= {P2_guess_list[i][1]})?"

        play_by_play.append(
            f"CPU Guess #{i + 1}: Is your number between {P2_guess_list[i][0]} and {P2_guess_list[i][1]} inclusive? <br> (i.e. {P2_guess_list[i][0]} <= x <= {P2_guess_list[i][1]})?"
        )

        if P2_guess_list[i][0] <= P2_get_secret_char() <= P2_guess_list[i][1]:
            computers_latest_guess += "<br>**Answer:** YES!"
            play_by_play.append(f"  Answer: YES!")
            for j in range(1, P2_guess_list[i][0]):
                P2_eliminated.add(j)
            for j in range(P2_guess_list[i][1] + 1, P2_max_chars_slider.value + 1):
                P2_eliminated.add(j)
        else:
            computers_latest_guess += "<br>**Answer:** NO!"
            play_by_play.append(f"  Answer: NO!")
            for j in range(P2_guess_list[i][0], P2_guess_list[i][1] + 1):
                P2_eliminated.add(j)

        play_by_play.append(f"    Eliminated so far: {P2_eliminated}")
        P2_chars_remaining = P2_max_chars_slider.value - len(P2_eliminated)

        play_by_play.append(f"    Num options remaining: {P2_chars_remaining}")
        History.append((P1_chars_remaining, P2_chars_remaining))

        if P2_chars_remaining == 1:
            play_by_play.append("    THE CPU WINS!")
            break

    computer_wins = P2_chars_remaining == 1
    computer_wins_text = " (WINNER!)" if computer_wins else ""
    computer_wins_kind = "success" if computer_wins else "neutral"

    human_wins = P1_chars_remaining == 1
    human_wins_text = " (WINNER!)" if human_wins else ""
    human_wins_kind = "success" if human_wins else "neutral"


    mo.vstack(
        [
            restart_button,
            mo.hstack(
                [
                    mo.callout(
                        mo.vstack(
                            [
                                mo.md("# Human" + human_wins_text),
                                mo.md(
                                    f"Your secret number is: {P2_get_secret_char()} "
                                ),
                                mo.md(
                                    f"## {P1_max_chars_slider.value - len(P1_eliminated)} Possibilities Remaining:"
                                ),
                                mo.md(
                                    ", ".join(
                                        map(
                                            str,
                                            sorted(
                                                n
                                                for n in range(
                                                    1,
                                                    P1_max_chars_slider.value + 1,
                                                )
                                                if n not in P1_eliminated
                                            ),
                                        )
                                    )
                                ),
                                mo.md("## Human's previous guess:"),
                                mo.md(humans_latest_guess),
                                mo.md("## Human's next guess:"),
                                mo.md("Slide to choose a range to guess about:"),
                                range_slider,
                                mo.md(
                                    f"**Question:** Is your number between {range_slider.value[0]} and {range_slider.value[1]} inclusive?<br>(i.e. {range_slider.value[0]} <= x <= {range_slider.value[1]})"
                                ),
                                guess_button,
                            ]
                        ),
                        kind=human_wins_kind,
                    ),
                    mo.callout(
                        mo.vstack(
                            [
                                mo.md("# Computer" + computer_wins_text),
                                mo.md(
                                    f"Secret number was {P1_get_secret_char()}"
                                    if computer_wins
                                    else f"Secret number is ???"
                                ),
                                mo.md(
                                    f"## {P2_max_chars_slider.value - len(P2_eliminated)} Possibilities Remaining:"
                                ),
                                mo.md(
                                    ", ".join(
                                        map(
                                            str,
                                            sorted(
                                                n
                                                for n in range(
                                                    1,
                                                    P2_max_chars_slider.value + 1,
                                                )
                                                if n not in P2_eliminated
                                            ),
                                        )
                                    )
                                ),
                                mo.md("## Computer's latest guess:"),
                                mo.md(computers_latest_guess),
                            ]
                        ),
                        kind=computer_wins_kind,
                    ),
                ]
            ),
        ]
    )
    return (
        CPU_guess_size,
        History,
        P1_chars_remaining,
        P2_chars_remaining,
        computer_wins,
        computer_wins_kind,
        computer_wins_text,
        computers_latest_guess,
        display_P1_eliminated,
        human_wins,
        human_wins_kind,
        human_wins_text,
        humans_latest_guess,
        i,
        j,
        play_by_play,
        smallest_non_eliminated,
    )


@app.cell
def _():
    return


@app.cell
def _(np):
    def compute_pstar_bstar(r):
        """
        Computes p*(n, m) and b*(n, m) for all 3 <= n+m <= r.

        Args:
            r (int): Upper bound, must satisfy r >= 3.

        Returns:
            p_star (ndarray): 2D array of shape (r+1, r+1) with p*(n, m)
            b_star (ndarray): 2D array of shape (r+1, r+1) with b*(n, m)
        """
        # Initialize arrays (index from 0 to r)
        p_star = np.zeros((r + 1, r + 1))
        b_star = np.zeros((r + 1, r + 1), dtype=int)

        # Base conditions:
        # p*(1, m) = 1 for all m > 1
        p_star[1, 2:] = 1.0
        # p*(n, 1) = 0 for all n > 1 (already 0 from initialization)

        # Main loop
        for s in range(3, r + 1):
            for n in range(1, s + 1):
                m = s - n
                if n < 1 or m < 1:
                    continue

                # Only compute for valid pairs (n, m)
                if n >= 2 and m > 1:
                    # Try all possible b in [1, n-1]
                    b_vals = np.arange(1, n // 2 + 1)
                    # Compute the objective for each b

                    values = (
                        1.0
                        - (b_vals / n) * p_star[m, b_vals]
                        - ((n - b_vals) / n) * p_star[m, n - b_vals]
                    )  # take the minimum!

                    # Find the maximizing b
                    epsilon = 0.00001
                    best_idx = np.argmax(values + epsilon * b_vals)
                    p_star[n, m] = values[best_idx]
                    b_star[n, m] = b_vals[best_idx]  # offset by 1

        return p_star, b_star


    r = 65
    p_star, b_star = compute_pstar_bstar(r)
    print("Compute Optimal p* and b* using the recursion")
    print("p* matrix:\n", np.round(p_star[1:12, 1:12], 2))
    print("b* matrix:\n", b_star[1:12, 1:12])
    return b_star, compute_pstar_bstar, p_star, r


@app.cell
def _(mo):
    regions_checkbox = mo.ui.checkbox(label="Show Regions")
    region_labels_checkbox = mo.ui.checkbox(label="Show Region Labels")
    history_checkbox = mo.ui.checkbox(label="Show Game History")
    radio_options = [
        "Show Optimal Bid",
        "Show Optimal Win Probability",
        "Show None",
    ]
    dropdown_options = ["Optimal Bid", "Optimal Win Probability", "Blank Plot"]
    plot_dropdown = mo.ui.dropdown(
        dropdown_options, value=dropdown_options[0], label="Plot:"
    )
    return (
        dropdown_options,
        history_checkbox,
        plot_dropdown,
        radio_options,
        region_labels_checkbox,
        regions_checkbox,
    )


if __name__ == "__main__":
    app.run()
