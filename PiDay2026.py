import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _(config):
    import marimo as mo
    from manim_slides import Slide
    import manim
    import numpy as np


    # This manually injects everything from manim into the local namespace
    # essentially does from manim impoort *, but in a way that marimo allows it
    for name in dir(manim):
        if not name.startswith("_"):
            globals()[name] = getattr(manim, name)

    config.quality = "high_quality"  # "low_quality"
    config.media_dir = "media"
    config.verbosity = "WARNING"
    config.progress_bar = "display"


    def render_scene(which_scene):
        scene = which_scene
        scene.render()

        # This dynamically retrieves the path to the movie file
        # that Manim just generated!
        return scene.renderer.file_writer.movie_file_path
    return Slide, manim, mo, name, np, render_scene


@app.cell
def _(
    Axes,
    BLUE,
    BarChart,
    Brace,
    Create,
    DEGREES,
    DOWN,
    DashedLine,
    FadeIn,
    FadeOut,
    GRAY_B,
    GREEN,
    GrowFromCenter,
    LEFT,
    LaggedStart,
    ManimColor,
    MathTex,
    MoveToTarget,
    RIGHT,
    ReplacementTransform,
    Slide,
    SurroundingRectangle,
    Transform,
    TransformMatchingTex,
    UL,
    UP,
    VGroup,
    Write,
    YELLOW,
    YELLOW_A,
    config,
    interpolate_color,
    np,
    rate_functions,
    smooth,
):
    # Global parameters
    n = 200  # Number of dimensions
    N_sims = 1000  # 0  # Number of simulations to perform
    bins_per_unit = 1
    N_bins = bins_per_unit * n  # Number of bins for the histogram

    np.random.seed(31415)  # Set random seed for reproducibility
    samples = np.random.binomial(n, 0.5, N_sims)  # , 1, (N_sims, n)) ** 2, axis=1)
    bins = np.linspace(0, n, N_bins)

    COLOR_1 = "#FF3333"
    COLOR_0 = "#0092CC"

    myYELLOW = "#DCD427"
    myGRAY = "#969696"
    # COLOR_PI = '#779933' #a green for pi

    # Global dictionary for TeX to color mapping

    n_color = BLUE
    non_var_color = GRAY_B
    t2cD = {
        r"{n}": n_color,
        r"{0}": COLOR_0,
        r"{1}": COLOR_1,
        r"Z": YELLOW,
        r"X": GREEN,
        r"\int": non_var_color,
        r"\mathbf{1}": YELLOW_A,
        r"\mathbb{P}": non_var_color,
        r"\mathbb{E}": non_var_color,
        r"\sigma^2": YELLOW,
        r"{k}": myYELLOW,
        r"{220,098}": myYELLOW,
        r"{5}": myYELLOW,
        r"{105}": myYELLOW,
        r"\mathcal{N}": YELLOW,
        r"\big(": non_var_color,
        r"\big)": non_var_color,
        r"\bigg(": non_var_color,
        r"\bigg)": non_var_color,
        r"\bigg)^": non_var_color,
        r"\Big]": non_var_color,
        r"\Big[": non_var_color,
        r"\Big)": non_var_color,
        r"\Big(": non_var_color,
    }

    my_zoom_scale_factor = 3.8


    def shrink_to_target(mover, target, run_time=1.5, rate_func=smooth):
        """
        Animates mover_mob moving to target_mob's center while shrinking to a point.
        """
        return (
            mover.animate(run_time=run_time, rate_func=rate_func)
            .move_to(target.get_center())
            .scale(0.01)
        )  # .set_opactiy(0)


    class PiDay(Slide):  # Scene):  # , Slide):
        def pause(self):
            self.wait(0.1)
            self.next_slide()

        def construct(self):
            print("Constructing...")
            self.wait(0.1)
            self.pause()
            self.wait(0.1)
            self.next_section()  # skip_animations=True)

            # --- Configuration ---

            def generate_binary_paragraph_tex_lines(num_lines, line_length):
                lines = []

                # Generate lines of binary text
                for _ in range(num_lines):
                    if _ != num_lines - 1:
                        line = "".join(
                            str(np.random.choice([0, 1]))
                            for _ in range(line_length)
                        )
                    elif _ == num_lines - 1:
                        line = "".join(
                            str(np.random.choice([0, 1]))
                            for _ in range(line_length - 3)
                        )
                        line = line + r"\ldots"

                    lines.append(line)

                return lines

            # Number of lines and line length
            num_lines, line_length = 5, 40  # Adjust as needed

            # Generate the lines of binary text and \ldots
            binary_paragraph_lines = generate_binary_paragraph_tex_lines(
                num_lines, line_length
            )

            # Specify the first three digits you want
            first_three_digits = "010"

            # Modify the first element of binary_paragraph_lines
            binary_paragraph_lines[0] = (
                first_three_digits + binary_paragraph_lines[0][3:]
            )

            # Specify the last three digits you want before "\ldots"
            last_three_digits = "011"

            # Modify the last element of binary_paragraph_lines
            binary_paragraph_lines[-1] = (
                binary_paragraph_lines[-1][: -len(r"\ldots") - 3]
                + last_three_digits
                + r"\ldots"
            )

            # first_three_digits = binary_paragraph_lines[0][0:3]
            # last_three_digits = binary_paragraph_lines[-1][
            #    -len(r"\ldots") - 3 : -len(r"\ldots")
            # ]
            print(first_three_digits, last_three_digits)

            # Create a VGroup of MathTex objects
            my_fs = 24
            colored_paragraph = VGroup(
                *[
                    MathTex(
                        line,
                        tex_to_color_map={"0": COLOR_0, "1": COLOR_1},
                        font_size=my_fs,
                    )
                    for line in binary_paragraph_lines
                ]
            )

            dots = MathTex(r"\vdots", font_size=my_fs)

            # Arrange the lines vertically
            paragraph = VGroup(
                *[colored_paragraph[i] for i in range(num_lines - 1)],
                # dots,
                colored_paragraph[-1],
            ).arrange(DOWN, buff=0.07)
            # Scale and center the paragraph
            screen_width_fraction = 0.95  # Make the width 80% of the screen width
            paragraph.width = config.frame_width * screen_width_fraction
            # colored_paragraph.move_to(ORIGIN)

            title_label = MathTex(
                r"\text{One Billion Random }",
                r"{0}",
                r"\text{'s and }",
                r"{1}",
                r"\text{'s}",
                tex_to_color_map=t2cD,
                font_size=52,
            ).to_edge(UP)
            paragraph.next_to(title_label, DOWN)

            # Show the paragraph in a scene
            self.add(title_label)
            self.wait(0.1)
            self.pause()

            # Show the paragraph with staggered animation
            self.play(
                LaggedStart(*[Write(line) for line in paragraph], lag_ratio=0.5)
            )
            # self.add(colored_paragraph)
            self.pause()

            my_fs = 90
            pi_approx = MathTex(
                r"\pi \approx",
                font_size=my_fs,
            )
            digits = "1554656"
            ans = MathTex(
                r"3.",
                *[digits[i] for i in range(num_lines - 1)],
                font_size=my_fs,
                tex_to_color_map=t2cD,
            )

            pi_group_buff = 1.8

            pi_group = (
                VGroup(pi_approx, ans)
                .arrange(RIGHT, buff=0.2)
                .to_edge(DOWN, buff=pi_group_buff)
            )

            self.play(Write(pi_approx))
            self.pause()

            self.play(
                LaggedStart(
                    *[
                        ReplacementTransform(colored_paragraph[i].copy(), ans[i])
                        for i in range(num_lines)
                    ],
                    lag_ratio=0.5,
                )
            )
            # self.add(colored_paragraph)
            self.pause()

            digits2 = "141234"
            ans2 = MathTex(
                r"3.",
                *[digits2[i] for i in range(num_lines - 1)],
                font_size=my_fs,
                tex_to_color_map=t2cD,
            )

            pi_group2 = (
                VGroup(pi_approx.copy(), ans2)
                .arrange(RIGHT, buff=0.2)
                .to_edge(DOWN, buff=pi_group_buff)
            )
            self.add(pi_group2[0])
            pi_group.generate_target()
            pi_group.target.scale(0.6)
            pi_group.target.to_edge(DOWN)

            self.play(MoveToTarget(pi_group))
            self.pause()

            self.play(
                LaggedStart(
                    *[
                        ReplacementTransform(colored_paragraph[i].copy(), ans2[i])
                        for i in range(num_lines)
                    ],
                    lag_ratio=0.5,
                )
            )
            # self.add(colored_paragraph)
            self.pause()

            # Define the number of digits per group and how many you want to show visibly
            # Define the number of digits per group (conceptually) and visible digits
            digits_per_group = 200  # This is the conceptual group size
            visible_digits = (
                8  # Number of digits visibly shown on each side of \ldots
            )

            def create_compressed_digit_group(digits_per_group, visible_digits):
                side_digits = (visible_digits - 1) // 2
                return (
                    "".join(
                        str(np.random.choice([0, 1])) for _ in range(side_digits)
                    )
                    + r"\ldots"
                    + "".join(
                        str(np.random.choice([0, 1])) for _ in range(side_digits)
                    )
                )

            # Create braces and labels
            top_brace_pre = Brace(
                paragraph, UP, color=myGRAY
            )  # Brace(digit_groups_column, UP)
            top_brace_label = MathTex(
                r"200 \text{ digits}", tex_to_color_map=t2cD, font_size=42
            ).next_to(top_brace_pre, UP, buff=0.1)

            s_rect_pre = SurroundingRectangle(paragraph, color=myYELLOW, buff=0.1)
            self.play(
                Create(s_rect_pre), FadeOut(title_label, pi_group, pi_group2)
            )
            self.play(Write(top_brace_label), Create(top_brace_pre))
            self.pause()

            # Create the groups of 200 random digits as compressed visible digits
            num_example_groups = 5  # Examples before and after \vdots
            compressed_digit_lines = [
                first_three_digits + r"\ldots" + last_three_digits
            ] + [
                create_compressed_digit_group(digits_per_group, visible_digits)
                for _ in range(num_example_groups)
            ]

            # Create a VGroup of MathTex objects for the compressed digit groups
            compressed_digit_groups = VGroup(
                *[
                    MathTex(
                        line,
                        tex_to_color_map={"0": COLOR_0, "1": COLOR_1},
                        font_size=my_fs,
                    )
                    for line in compressed_digit_lines
                ]
            )

            # Add a \vdots between the groups to show continuation
            vdots = MathTex(r"\vdots", font_size=my_fs)

            # Arrange the digit groups vertically. Place \vdots as the second to last row.
            digit_groups_column = VGroup(
                *[mob for mob in compressed_digit_groups[0:-1]],
                vdots,
                compressed_digit_groups[-1],
            ).arrange(DOWN, buff=0.15)
            digit_groups_column[-2].shift(0.25 * DOWN)

            # Adjust the width and position of the column
            digit_groups_column.width = paragraph.width / 5
            digit_groups_column.align_to(paragraph, UP)

            s_rect_post = SurroundingRectangle(
                digit_groups_column[0], color=myYELLOW, buff=0.1
            )
            top_brace_post = Brace(
                digit_groups_column[0], UP, color=myGRAY
            )  # Brace(digit_groups_column, UP)

            # Transform paragraph to digit groups column
            self.play(
                ReplacementTransform(paragraph, digit_groups_column[0]),
                ReplacementTransform(s_rect_pre, s_rect_post),
                ReplacementTransform(top_brace_pre, top_brace_post),
            )
            self.pause()

            left_brace = Brace(digit_groups_column, LEFT, color=myGRAY)
            left_brace_label = MathTex(
                r"\text{5 million groups}",
                tex_to_color_map=t2cD,
                font_size=40,
            ).next_to(left_brace, LEFT, buff=0.1)

            # Animate braces and labels
            # self.play(
            #    GrowFromCenter(top_brace),
            ##    Write(top_brace_label),
            # )
            # self.pause()

            # Transform paragraph to digit groups column
            self.play(
                FadeOut(s_rect_post),
                LaggedStart(
                    GrowFromCenter(left_brace),
                    Write(left_brace_label),
                    *[FadeIn(mob, shift=DOWN) for mob in digit_groups_column[1:]],
                    lag_ratio=0.2,
                ),
            )
            self.pause()

            my_scale = 0.8

            left_brace_label.generate_target()
            left_brace_label.target.rotate(90 * DEGREES).scale(my_scale)

            my_stuff = VGroup(
                top_brace_post, top_brace_label, left_brace, digit_groups_column
            )  # left_brace_label is missing!
            my_stuff.generate_target()

            my_stuff.target.scale(my_scale).to_corner(UL, buff=0.3).shift(
                left_brace_label.target.width * RIGHT
            )
            left_brace_label.target.next_to(my_stuff.target, LEFT, buff=0.1).shift(
                0.4 * DOWN
            )

            self.play(MoveToTarget(my_stuff), MoveToTarget(left_brace_label))
            self.pause()

            my_stuff += left_brace_label

            # self.play(
            # )
            # self.pause()

            # --- Histogram of 1's ---

            def my_hist_values(_samples):
                hist_values = np.zeros(N_bins)
                for sample in _samples:
                    bin_index = np.digitize(sample, bins) - 1
                    bin_index = min(bin_index, len(hist_values) - 1)
                    hist_values[bin_index] += 1
                return hist_values

            complete_hist_values = my_hist_values(samples)
            max_hist_value = np.max(complete_hist_values)

            HIST_HEIGHT = 4.5
            HIST_WIDTH = 8

            bar_colors = []
            diff = 10
            blue_edge = 100 - diff
            red_edge = 100 + diff

            low_color = ManimColor(COLOR_0)
            high_color = ManimColor(COLOR_1)
            some_color = myGRAY
            for i in range(N_bins - 1):
                if i < blue_edge:
                    color = low_color
                elif i > red_edge:
                    color = high_color
                else:
                    interpolation_factor = (i - blue_edge) / (red_edge - blue_edge)
                    color = interpolate_color(
                        low_color, high_color, interpolation_factor
                    )
                bar_colors.append(color)

            def my_bar_chart(_hist_values, scale_factor=1.0):
                max_val = max_hist_value  # max(np.max(_hist_values),1)
                chart = BarChart(
                    values=_hist_values,  # np.minimum(_hist_values*scale_factor, max_hist_value),
                    y_range=[
                        0,
                        max_val,
                        max_val,
                    ],  # [0, max_hist_value, max_hist_value],
                    bar_colors=bar_colors,
                    y_length=HIST_HEIGHT,
                    x_length=HIST_WIDTH,
                    x_axis_config={
                        "font_size": 24,
                        "stroke_color": some_color,
                        "stroke_opacity": 0,
                    },
                    y_axis_config={
                        "font_size": 1,
                        "stroke_color": some_color,
                        "stroke_opacity": 0,
                    },
                )
                chart.to_edge(RIGHT, buff=1.5)
                chart.to_edge(UP, buff=1.0)
                return chart

            hist_values = np.zeros(N_bins)
            bar_chart = my_bar_chart(hist_values)

            # Create and add text objects for the initial display
            label_fs = 36
            down_len = 0.3
            label_left = MathTex(
                r"0", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[0].get_center() + DOWN * down_len)

            label_right = MathTex(
                r"{200}", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[-1].get_center() + DOWN * down_len)

            label_mid = MathTex(
                r"100", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[N_bins // 2].get_center() + DOWN * down_len)

            label = MathTex(
                r"\text{Histogram: Number of }{1}\text{'s in 200 digits}",
                tex_to_color_map=t2cD,
            )
            label.width = bar_chart.width
            label.next_to(bar_chart, UP)
            # uline = Underline(label, color=myGRAY)
            # label = VGroup(label, uline)

            sample_fs = 40
            # sample_count_text = MathTex(
            #    r"\text{Samples:} 0",
            #    font_size=sample_fs,
            #    color=myGRAY,
            # ).next_to(bar_chart, DOWN)

            def my_samples_label(m):
                return (
                    MathTex(
                        r"\text{Samples: } ",
                        str(m),
                        font_size=sample_fs,
                        color=myGRAY,
                    )
                    .next_to(label, DOWN)
                    .align_to(label, RIGHT)
                )

            samples_label = my_samples_label(0)

            self.play(
                Create(bar_chart),
                Write(label_left),
                Write(label_right),
                Write(label_mid),
                Write(label),
                Write(samples_label),
            )
            self.pause()

            self.play(
                ReplacementTransform(
                    digit_groups_column.copy(), bar_chart.bars[100]
                )
            )

            def my_new_label(m):
                return (
                    MathTex(
                        r"\text{Samples: } ",
                        "{:,}".format(m),
                        font_size=sample_fs,
                        color=myGRAY,
                    )
                    .next_to(label, DOWN)
                    .align_to(label, RIGHT)
                )

            new_label = my_samples_label(0)
            N_updates = 50  # 25
            N_samples_to_update = int(len(samples) / N_updates)  # 100
            for i in range(0, len(samples), N_samples_to_update):
                current_samples = samples[i : i + N_samples_to_update]
                my_scale_factor = min(5, (len(samples) / (i + 1)) ** 0.5)
                hist_values += my_hist_values(current_samples)
                new_chart = my_bar_chart(hist_values, my_scale_factor)

                new_label = my_new_label(i + N_samples_to_update)

                self.play(
                    Transform(bar_chart.bars, new_chart.bars),
                    Transform(samples_label, new_label),
                    run_time=0.1,
                    rate_func=rate_functions.linear,
                )

            self.pause()

            # Create and animate the dashed line with label
            # Access the x and y axes directly from the bar chart
            x_axis = bar_chart.x_axis
            y_axis = bar_chart.y_axis

            # Calculate the position for x = n/3

            # Find the bin with the maximum height
            max_bin_index = np.argmax(hist_values)

            # Calculate the position for the center of the max bin
            max_bin_position = x_axis.number_to_point(100 + 0.5)[0]
            dashed_line = DashedLine(
                start=[max_bin_position, y_axis.number_to_point(0)[1], 0],
                end=[
                    max_bin_position,
                    y_axis.number_to_point(max_hist_value)[1],
                    0,
                ],
                color=myYELLOW,
                dash_length=0.1,
            )

            self.play(
                Create(dashed_line),  # Write(n_over_3_label)
            )  # Create and animate the dashed line with label
            self.pause()

            # n_over_3_position = x_axis.number_to_point(n/3*bins_per_unit)[0]
            ##dashed_line = DashedLine(
            #    start=[n_over_3_position, y_axis.number_to_point(0)[1], 0],
            #    end=[n_over_3_position, y_axis.number_to_point(max_hist_value)[1], 0],
            #    color=YELLOW,
            #    dash_length=0.1,
            # )

            # n_over_3_label = MathTex(
            #    r"", tex_to_color_map=t2cD, font_size=label_fs
            # ).next_to(dashed_line, DOWN, buff=0)
            # n_over_3_label.shift(DOWN * 0.1)

            ax = Axes(
                x_range=[0, n, 0.1],
                y_range=[0, 1, 0.1],
                tips=False,
                x_length=HIST_WIDTH,
                y_length=HIST_HEIGHT,
                axis_config={"include_numbers": False},
            )

            # x_min must be > 0 because log is undefined at 0.
            HIST_COLOR = GREEN
            Z_COLOR = myYELLOW
            # Make the Bell curve

            off_x = (100 + 0.5) / bins_per_unit

            sanity_plot = ax.plot(
                lambda x: np.exp(-((x - off_x) ** 2)),
                x_range=[0, n],
                color=HIST_COLOR,
            )
            graph = ax.plot(
                lambda x: np.exp(-((x - off_x) ** 2) / (0.5 * n)),
                x_range=[0, n],
                stroke_width=5,
                color=Z_COLOR,
                use_smoothing=True,
                stroke_opacity=0.65,
            )
            area = ax.get_area(
                graph,
                x_range=(0, n),
                color=Z_COLOR,  # HIST_COLOR,
                opacity=0.1,
            )
            graph.height = sanity_plot.height
            area.height = sanity_plot.height
            # BELL_CURVE = VGroup(ax, graph, area)
            VGroup(graph, area).align_to(bar_chart, UP).align_to(bar_chart, RIGHT)

            # ax, graph, area = BELL_CURVE
            self.play(Create(graph), run_time=2)  # FadeIn(area),
            self.pause()

            # return 0

            label_spacing = 0.7

            # Create and animate the dashed line with label
            CLT_label_1 = MathTex(
                r"\text{Bell Curve Function}",
                tex_to_color_map=t2cD,
                font_size=label_fs * 1.2,
            )
            CLT_label_2 = MathTex(
                r"=\frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2}",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                # color='#B7B327',
                font_size=label_fs * 1.2,
            )

            CLT_label = (
                VGroup(CLT_label_1, CLT_label_2)
                .arrange(DOWN, buff=0.1)
                .next_to(dashed_line, LEFT, buff=label_spacing)
            )

            self.play(Write(CLT_label_1))
            self.play(Write(CLT_label_2))
            self.pause()

            # Create and animate the dashed line with label
            bin_label_fs = label_fs * 1.2
            bin_label_1 = MathTex(
                r"\mathbb{P}\big(\text{Exactly }",
                r"100",
                r"\;",
                r"{1}\text{'s}\big)",
                tex_to_color_map=t2cD,
                font_size=bin_label_fs,
            )

            # bin_label_2 = MathTex(
            #    r"= \frac{1}{2^{100}}\binom{200}{100}",
            # r"\! + \!",
            # r"\sqrt{ {n} }"
            # r"\frac{ 2 }{ 3 \sqrt{5} }",
            # r"Z",
            #    font_size=label_fs,
            # )
            bin_label_2 = MathTex(
                r"\approx \frac{1}{10\sqrt{\pi}}",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                font_size=label_fs * 1.3,
            )

            bin_label = (
                VGroup(bin_label_1, bin_label_2)  # , bin_label_3)
                .arrange(DOWN, buff=0.1)  # , aligned_edge=LEFT
                .next_to(dashed_line, RIGHT, buff=label_spacing)
            )

            # self.play(Write(bin_label_1))
            # self.play(Write(bin_label_2))

            self.play(
                ReplacementTransform(CLT_label_2.copy(), bin_label_2),
                ReplacementTransform(CLT_label_1.copy(), bin_label_1),
            )

            self.pause()

            # Create and animate the dashed line with label
            approx_label_1 = MathTex(
                r"{ {\text{Number with exactly 100 }{1}\text{'s}}",
                r"\over",
                r"{\text{Total number of samples}}",
                tex_to_color_map=t2cD,
                font_size=label_fs * 0.9,
            )
            print(*enumerate(approx_label_1), sep="\n")

            # bin_label_2 = MathTex(
            #    r"= \frac{1}{2^{100}}\binom{200}{100}",
            # r"\! + \!",
            # r"\sqrt{ {n} }"
            # r"\frac{ 2 }{ 3 \sqrt{5} }",
            # r"Z",
            #    font_size=label_fs,
            # )
            approx_label_2 = MathTex(
                r"\approx \frac{1}{10\sqrt{\pi}}",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                font_size=label_fs * 1.3,
            )

            approx_label = (
                VGroup(approx_label_1, approx_label_2)  # , bin_label_3)
                .arrange(DOWN, buff=0.1)
                .next_to(dashed_line, LEFT, buff=label_spacing)
            )

            self.play(
                FadeOut(CLT_label),
                TransformMatchingTex(
                    bin_label_2.copy(),
                    approx_label_2,
                    replace_mobject_with_target_in_scene=True,
                ),
                ReplacementTransform(bin_label_1.copy(), approx_label_1),
            )
            self.pause()

            def find_index_of_entry(math_tex_object, string_to_find):
                """
                Find the index of the entry in a MathTex object that matches the given string.

                Parameters:
                - math_tex_object (MathTex): The MathTex object to search through.
                - string_to_find (str): The LaTeX string to find.

                Returns:
                - int: The index of the entry that matches the string, or -1 if not found.
                """
                # Iterate through each subobject in the MathTex
                for index, submobject in enumerate(math_tex_object):
                    if submobject.get_tex_string() == string_to_find:
                        return index
                return -1

            numerator = MathTex("281,474", font_size=label_fs * 1.15)
            denominator = MathTex("5,000,000", font_size=label_fs * 1.15)
            ix_of_over = find_index_of_entry(approx_label_1, r"\over")
            numerator.move_to(approx_label_1[0:ix_of_over])
            denominator.move_to(approx_label_1[ix_of_over + 1 :])

            self.play(
                FadeOut(approx_label_1[ix_of_over + 1 :]),
                ReplacementTransform(left_brace_label.copy(), denominator),
            )
            self.play(
                FadeOut(approx_label_1[0:ix_of_over]),
                approx_label_1[ix_of_over].animate.scale(0.5),
                ReplacementTransform(digit_groups_column.copy(), numerator),
            )
            self.pause()

            my_fs = 50
            my_pi_approx = MathTex(
                r"\pi",
                r"\approx",
                r"\bigg({5,000,000 \over 10 \times",
                r"281,474",
                r"}",
                r"\bigg)^2",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                tex_to_color_map=t2cD,
                font_size=my_fs,
            )
            ans = MathTex(r"\approx 3.1554", font_size=my_fs)
            my_pi_group = (
                VGroup(my_pi_approx, ans)
                .arrange(RIGHT, buff=0.2)
                .to_edge(DOWN, buff=0.4)
            )  # , buff=pi_group_buff)

            self.play(
                ReplacementTransform(
                    VGroup(numerator, denominator, approx_label_2).copy(),
                    my_pi_approx,
                )
            )
            self.play(Write(ans))
            # self.pause()
            self.next_slide(loop=True)

            ## end of classic method

            new_bin_label_1 = MathTex(
                r"\mathbb{P}\big(\text{Exactly }",
                r"{105}",
                r"\;",
                r"{1}\text{'s}\big)",
                tex_to_color_map=t2cD,
                font_size=bin_label_fs,
            )
            new_bin_label_1.align_to(bin_label_1, DOWN)
            new_bin_label_1.align_to(bin_label_1, LEFT)

            # bin_label_2 = MathTex(
            #    r"= \frac{1}{2^{100}}\binom{200}{100}",
            # r"\! + \!",
            # r"\sqrt{ {n} }"
            # r"\frac{ 2 }{ 3 \sqrt{5} }",
            # r"Z",
            #    font_size=label_fs,
            # )

            correction_term = MathTex(
                r"e^{- {5}^2/100}",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                tex_to_color_map=t2cD,
                font_size=label_fs * 1.3,
            )
            correction_term.next_to(bin_label_2, RIGHT, buff=0.15).shift(0.2 * UP)

            old_x = dashed_line.get_x()
            new_x = x_axis.number_to_point(105 + 0.5)[0]
            self.play(
                dashed_line.animate.shift(LEFT), rate_func=rate_functions.wiggle
            )

            self.pause()

            self.play(
                dashed_line.animate.shift((new_x - old_x) * RIGHT),
                rate_func=rate_functions.ease_in_out_back,
            )
            my105 = (
                MathTex(r"105", font_size=label_fs, color=myYELLOW)
                .next_to(dashed_line, DOWN)
                .align_to(label_mid, DOWN)
            )
            label_mid_copy = label_mid.copy()
            self.play(ReplacementTransform(label_mid, my105))
            self.pause()

            self.play(
                LaggedStart(
                    shrink_to_target(my105.copy(), bin_label_1),
                    TransformMatchingTex(bin_label_1, new_bin_label_1),
                    lag_ratio=0.5,
                )
            )
            self.play(Write(correction_term))
            self.pause()

            numerator2 = MathTex(
                "220,098", font_size=label_fs * 1.15, color=myYELLOW
            )
            numerator2.move_to(numerator)
            approx_label_2.generate_target()
            approx_label_2.target.shift(0.5 * LEFT)
            correction_term_copy = (
                correction_term.copy()
                .next_to(approx_label_2.target, RIGHT, buff=0.15)
                .shift(0.2 * UP)
            )

            self.play(
                FadeOut(numerator),
                ReplacementTransform(digit_groups_column.copy(), numerator2),
                ReplacementTransform(correction_term.copy(), correction_term_copy),
                MoveToTarget(approx_label_2),
            )
            self.pause()

            my_pi_approx2 = MathTex(
                r"\pi",
                r"\approx",
                r"\bigg({5,000,000 \over 10 \times",
                r"{220,098}",
                r"}",
                r"e^{- {5}^2/100}",
                r"\bigg)^2",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                tex_to_color_map=t2cD,
                font_size=my_fs,
            )

            my_pi_approx2.move_to(my_pi_approx)
            my_pi_approx2.align_to(my_pi_approx, LEFT)

            ans2 = MathTex(r"\approx", r"3.1301", font_size=my_fs).next_to(
                my_pi_approx2, buff=0.2
            )
            self.play(
                FadeOut(ans),
                LaggedStart(
                    shrink_to_target(
                        VGroup(numerator2, correction_term_copy).copy(),
                        my_pi_approx,
                    ),
                    TransformMatchingTex(my_pi_approx, my_pi_approx2),
                    lag_ratio=0.5,
                ),
            )
            self.pause()

            self.play(FadeIn(ans2, shift=RIGHT))
            self.pause()

            pi_approx_only = MathTex(
                r"\pi",
                r"\approx",
                font_size=my_fs,
            )

            new_pi_formula = MathTex(
                r"{",
                r"3.1554",
                r"+",
                r"3.1301",
                r"\over",
                r"2",
                r"}",
                font_size=my_fs,
            )

            new_pi = VGroup(pi_approx_only, new_pi_formula).arrange(
                RIGHT, buff=0.2
            )

            new_pi.move_to(numerator2)
            new_pi.align_to(my_pi_approx2, LEFT)

            self.play(
                TransformMatchingTex(my_pi_approx2.copy(), pi_approx_only),
                TransformMatchingTex(ans2.copy(), new_pi_formula),
                FadeOut(
                    numerator2,
                    denominator,
                    correction_term_copy,
                    approx_label_2,
                    approx_label_1[ix_of_over],
                ),
            )
            self.pause()

            avg_equals = MathTex(r"\approx 3.14275", font_size=my_fs)
            avg_equals.next_to(new_pi, DOWN)
            avg_equals.align_to(pi_approx_only[1], LEFT)
            self.play(FadeIn(avg_equals, shift=DOWN))
            # self.pause()
            self.next_slide(loop=True)
            final_bin_label_1 = MathTex(
                r"\mathbb{P}\big(\text{Exactly }",
                r"100",
                r"\!+\!",
                r"{k}",
                r"\;",
                r"{1}\text{'s}\big)",
                tex_to_color_map=t2cD,
                font_size=bin_label_fs,
            )
            final_bin_label_1.align_to(new_bin_label_1, DOWN)
            final_bin_label_1.align_to(new_bin_label_1, LEFT)

            # bin_label_2 = MathTex(
            #    r"= \frac{1}{2^{100}}\binom{200}{100}",
            # r"\! + \!",
            # r"\sqrt{ {n} }"
            # r"\frac{ 2 }{ 3 \sqrt{5} }",
            # r"Z",
            #    font_size=label_fs,
            # )

            final_correction_term = MathTex(
                r"e^{- {k}^2/100}",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                tex_to_color_map=t2cD,
                font_size=label_fs * 1.3,
            )
            final_correction_term.move_to(correction_term).align_to(
                correction_term, LEFT
            )

            final_my105 = MathTex(
                r"100",
                r"\!+\!",
                r"{k}",
                font_size=label_fs,
                tex_to_color_map=t2cD,
            ).move_to(label_mid_copy)

            self.play(
                dashed_line.animate.shift(LEFT), rate_func=rate_functions.wiggle
            )
            self.pause()  # loop=True)

            self.play(
                TransformMatchingTex(my105, final_my105, key_map={"105": r"100"})
            )
            self.pause()

            self.play(
                TransformMatchingTex(
                    new_bin_label_1, final_bin_label_1, key_map={"{5}": r"{k}"}
                )
            )
            self.play(
                TransformMatchingTex(
                    correction_term, final_correction_term, key_map={"{5}": r"{k}"}
                )
            )
            self.pause()

            final_pi_formula = MathTex(
                r"{",
                r"3.1554",
                r"\!",
                r"+",
                r"\!",
                r"\ldots",
                r"\!",
                r"+",
                r"\!",
                r"3.1301",
                r"\over",
                r"21",
                r"}",
                r"{}",
                font_size=my_fs,
            ).scale(0.8)

            def my_dashed_line(ix):
                max_bin_position = x_axis.number_to_point(ix + 0.5)[0]
                dashed_line = DashedLine(
                    start=[max_bin_position, y_axis.number_to_point(0)[1], 0],
                    end=[
                        max_bin_position,
                        y_axis.number_to_point(max_hist_value)[1],
                        0,
                    ],
                    color=myYELLOW,
                    dash_length=0.1,
                )
                dashed_line.opacity = 0.7

                return dashed_line

            many_lines = [my_dashed_line(ix) for ix in [90, 95, 100, 110]]
            self.play(
                LaggedStart(*[Create(mob) for mob in many_lines], lag_ratio=0.5)
            )
            self.pause()

            final_pi_formula.move_to(new_pi_formula)
            final_pi_formula.align_to(new_pi_formula, LEFT)
            lines_copy = VGroup(many_lines).copy()
            digit_groups_column_copy = digit_groups_column.copy()
            self.play(
                shrink_to_target(digit_groups_column_copy, pi_approx_only),
                shrink_to_target(lines_copy, pi_approx_only),
                FadeOut(avg_equals),
            )
            self.remove(lines_copy)
            self.remove(digit_groups_column_copy)
            self.play(
                TransformMatchingTex(
                    new_pi_formula,
                    final_pi_formula,
                    key_map={r"2": r"21"},
                )
            )
            self.pause()

            final_avg_equals = MathTex(r"\approx 3.1412", font_size=my_fs).scale(2)

            final_avg_equals.next_to(final_pi_formula, DOWN, buff=0.6)
            final_avg_equals.align_to(pi_approx_only[1], LEFT)
            self.play(FadeIn(final_avg_equals, shift=DOWN))
            surrounding_rectangle = SurroundingRectangle(
                final_avg_equals,
                color=myYELLOW,
                buff=0.2,
            )
            self.play(Create(surrounding_rectangle))
            self.pause()

            self.wait(1)
            print("Done!")
    return (
        COLOR_0,
        COLOR_1,
        N_bins,
        N_sims,
        PiDay,
        bins,
        bins_per_unit,
        myGRAY,
        myYELLOW,
        my_zoom_scale_factor,
        n,
        n_color,
        non_var_color,
        samples,
        shrink_to_target,
        t2cD,
    )


@app.cell
def _(PiDay, mo, render_scene):
    _video_path = render_scene(PiDay())
    mo.video(_video_path, loop=True, controls=True, autoplay=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
