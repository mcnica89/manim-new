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

    config.quality = "low_quality"
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
    Scene,
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
):
    # Global parameters
    n = 200  # Number of dimensions
    N_sims = 5000  # 0  # Number of simulations to perform
    bins_per_unit = 1
    N_bins = bins_per_unit * n  # Number of bins for the histogram

    np.random.seed(42)  # Set random seed for reproducibility
    samples = np.random.binomial(n, 0.5, N_sims)  # , 1, (N_sims, n)) ** 2, axis=1)
    bins = np.linspace(0, n, N_bins)

    COLOR_1 = "#FF3333"
    COLOR_0 = "#0092CC"
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
        r"\mathcal{N}": YELLOW,
        r"\bigg(": non_var_color,
        r"\bigg)": non_var_color,
        r"\bigg)^": non_var_color,
        r"\Big]": non_var_color,
        r"\Big[": non_var_color,
        r"\Big)": non_var_color,
        r"\Big(": non_var_color,
    }

    my_zoom_scale_factor = 3.8


    class PiDay(Scene):  # , Slide):
        def pause(self):
            self.wait(0.1)
            # self.next_slide()

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

            myYELLOW = "#DCD427"
            myGRAY = "#969696"

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
            compressed_digit_lines = ["101\ldots000"] + [
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
                r"\text{5 million total groups}",
                tex_to_color_map=t2cD,
                font_size=36,
            ).next_to(left_brace, LEFT, buff=0.1)

            self.pause()

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

            def my_bar_chart(_hist_values):
                chart = BarChart(
                    values=_hist_values,
                    y_range=[0, max_hist_value, max_hist_value],
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
                r"\text{Histogram of Number of }{1}\text{'s in our Samples}",
                tex_to_color_map=t2cD,
            )
            label.width = bar_chart.width
            label.next_to(bar_chart, UP)
            # uline = Underline(label, color=myGRAY)
            # label = VGroup(label, uline)

            sample_fs = 40
            sample_count_text = MathTex(
                r"\text{Samples:} 0",
                font_size=sample_fs,
                color=myGRAY,
            ).next_to(bar_chart, DOWN)

            self.play(
                Create(bar_chart),
                Write(label_left),
                Write(label_right),
                Write(label_mid),
                Write(label),
            )
            self.pause()

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
            self.play(Write(samples_label))
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

            new_label = my_new_label(0)

            N_samples_to_update = 2_00  # 0  # 100
            for i in range(0, len(samples), N_samples_to_update):
                current_samples = samples[i : i + N_samples_to_update]
                hist_values += my_hist_values(current_samples)
                new_chart = my_bar_chart(hist_values)

                new_label = my_new_label(i + N_samples_to_update)

                self.play(
                    Transform(bar_chart, new_chart),
                    Transform(samples_label, new_label),
                    run_time=0.1,
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
            max_bin_position = x_axis.number_to_point(max_bin_index + 0.5)[0]
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

            off_x = (max_bin_index + 0.5) / bins_per_unit

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

            label_spacing = 0.7

            # Create and animate the dashed line with label
            CLT_label_1 = MathTex(
                r"\text{Bell Curve Function}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            CLT_label_2 = MathTex(
                r"=\frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}x^2}",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                # color='#B7B327',
                font_size=label_fs,
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
            bin_label_1 = MathTex(
                r"\mathbb{P}\bigg(\text{Exactly 100 }{1}\text{'s}\bigg)",
                tex_to_color_map=t2cD,
                font_size=label_fs,
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
                r"\bigg(\frac{5,000,000}{10 \times 281,474}\bigg)^2",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                font_size=my_fs,
            )
            ans = MathTex(r"\approx 3.1554", font_size=my_fs)
            my_pi_group = (
                VGroup(my_pi_approx, ans)
                .arrange(RIGHT, buff=0.2)
                .to_edge(DOWN, buff=0.6)
            )  # , buff=pi_group_buff)

            self.play(
                ReplacementTransform(
                    VGroup(numerator, denominator, approx_label_2).copy(),
                    my_pi_approx,
                )
            )
            self.play(Write(ans))
            self.pause()

            return 0

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
        my_zoom_scale_factor,
        n,
        n_color,
        non_var_color,
        samples,
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
