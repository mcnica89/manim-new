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
    Arrow,
    Axes,
    BLUE,
    BarChart,
    BraceBetweenPoints,
    Create,
    DOWN,
    DashedLine,
    GRAY_B,
    GRAY_C,
    GREEN,
    LEFT,
    MathTex,
    RED,
    RIGHT,
    Transform,
    UL,
    UP,
    VGroup,
    Write,
    YELLOW,
    YELLOW_A,
    ZoomedScene,
    np,
):
    # Global parameters
    n = 10  # Number of dimensions
    N_sims = 10_000  # Number of simulations to perform
    bins_per_unit = 5
    N_bins = bins_per_unit * n  # Number of bins for the histogram

    np.random.seed(42)  # Set random seed for reproducibility
    samples = np.sum(np.random.uniform(0, 1, (N_sims, n)) ** 2, axis=1)
    bins = np.linspace(0, n, N_bins)

    # Global dictionary for TeX to color mapping

    n_color = BLUE
    non_var_color = GRAY_B
    t2cD = {
        r"R": RED,
        r"{n}": n_color,
        r"{1}": n_color,
        r"{2}": n_color,
        r"Z": YELLOW,
        r"X": GREEN,
        r"\int": GRAY_B,
        r"\mathbf{1}": YELLOW_A,
        r"\mathbb{P}": non_var_color,
        r"\mathbb{E}": non_var_color,
        r"\sigma^2": YELLOW,
        r"\mathcal{N}": YELLOW,
        r"\bigg(": YELLOW,
        r"\bigg)": YELLOW,
        r"\bigg)^": non_var_color,
        r"\Big]": non_var_color,
        r"\Big[": non_var_color,
        r"\Big)": non_var_color,
        r"\Big(": non_var_color,
    }

    my_zoom_scale_factor = 3.8


    class ProbabilitySpheres(ZoomedScene):  # , Slide):
        def __init__(self, **kwargs):
            ZoomedScene.__init__(
                self,
                zoomed_display_corner=np.array([-1, 1, 0.0]),
                zoom_factor=0.23,
                zoomed_display_height=16 / my_zoom_scale_factor,
                zoomed_display_width=16 / my_zoom_scale_factor,
                zoomed_display_corner_buff=0.3,
                zoomed_camera_config={
                    "default_frame_stroke_width": 3,
                },
                **kwargs,
            )

        def pause(self):
            self.wait(0.1)
            # self.next_slide()

        def construct(self):
            print("Constructing...")
            self.wait(0.1)
            self.pause()
            self.wait(0.1)
            self.next_section(skip_animations=True)

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

            def my_bar_chart(_hist_values):
                low_color = RED
                high_color = BLUE
                some_color = GRAY_B
                chart = BarChart(
                    values=_hist_values,
                    y_range=[0, max_hist_value, max_hist_value],
                    bar_colors=[
                        low_color if i < bins_per_unit else high_color
                        for i in range(N_bins - 1)
                    ],
                    y_length=HIST_HEIGHT,
                    x_length=HIST_WIDTH,
                    x_axis_config={"font_size": 24, "stroke_color": some_color},
                    y_axis_config={"font_size": 1, "stroke_color": some_color},
                )
                chart.to_edge(RIGHT)
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
            label_left_2 = MathTex(
                r"1", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(
                bar_chart.bars[1 * bins_per_unit - 1].get_center()
                + DOWN * down_len
            )
            label_right = MathTex(
                r"{n}", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[-1].get_center() + DOWN * down_len)
            label_mid = MathTex(
                r"{n}/2", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[N_bins // 2].get_center() + DOWN * down_len)
            label = MathTex(
                r"\text{Histogram of } X^2_{1} + X^2_{2} + \ldots + X^2_{n}",
                tex_to_color_map=t2cD,
            ).next_to(bar_chart, UP)

            sample_fs = 40
            sample_count_text = MathTex(
                r"\text{Samples:} 0", font_size=sample_fs
            ).next_to(bar_chart, DOWN)

            self.play(
                Create(bar_chart),
                Write(label_left),
                Write(label_right),
                Write(label_mid),
                Write(label),
                Write(label_left_2),
            )
            self.pause()

            self.play(Create(bar_chart))
            self.pause()

            def my_samples_label(m):
                return (
                    MathTex(r"\text{Samples:} ", str(m), font_size=sample_fs)
                    .next_to(label, DOWN)
                    .align_to(label, RIGHT)
                )

            samples_label = my_samples_label(0)
            self.play(Write(samples_label))
            self.pause()

            def my_new_label(m):
                return (
                    MathTex(
                        r"\text{Samples:} ", "{:,}".format(m), font_size=sample_fs
                    )
                    .next_to(label, DOWN)
                    .align_to(label, RIGHT)
                )

            new_label = my_new_label(0)

            N_samples_to_update = 1000  # 100
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
                color=YELLOW,
                dash_length=0.1,
            )

            # n_over_3_position = x_axis.number_to_point(n/3*bins_per_unit)[0]
            ##dashed_line = DashedLine(
            #    start=[n_over_3_position, y_axis.number_to_point(0)[1], 0],
            #    end=[n_over_3_position, y_axis.number_to_point(max_hist_value)[1], 0],
            #    color=YELLOW,
            #    dash_length=0.1,
            # )

            n_over_3_label = MathTex(
                r"{n}/3", tex_to_color_map=t2cD, font_size=label_fs
            ).next_to(dashed_line, DOWN, buff=0)
            n_over_3_label.shift(DOWN * 0.1)

            # Create and animate the dashed line with label
            LLN_label_1 = MathTex(
                r"\text{Law of Large Numbers:}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            LLN_label_2 = MathTex(
                r"{{ X^2_{1} + \ldots + X^2_{n} }",
                r"\approx",
                r"{n}",
                r"\mathbb{E}",
                r"\Big[",
                r"X",
                r"^2",
                r"\Big]",
                r"=",
                r"{n}",
                r"{1 \over 3}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )

            LLN_eqn = MathTex(
                r"\mathbb{E}",
                r"\Big[",
                r"X",
                r"^2",
                r"\Big]",
                r"=",
                r"\int",
                r"_{-1}",
                r"^1",
                r"{x^2 \over 2}",
                r"dx",
                r"={",
                r"1 \over 3}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )

            my_buff = 0.4
            space_buff = 0.1
            LLN_label = (
                VGroup(LLN_label_1, LLN_label_2)
                .arrange(DOWN, buff=space_buff)
                .to_corner(UL, buff=my_buff)
            )

            LLN_eqn.next_to(LLN_label, DOWN).align_to(LLN_label, LEFT)

            self.play(
                Create(dashed_line), Write(n_over_3_label)
            )  # Create and animate the dashed line with label
            self.pause()

            self.play(Write(LLN_label_1))
            self.play(Write(LLN_label_2[:-3]))
            self.pause()

            self.play(Transform(LLN_label_2[-8:-3].copy(), LLN_eqn[0:5]))
            self.play(Write(LLN_eqn[5:]))
            self.pause()

            self.play(Transform(LLN_eqn[-2:].copy(), LLN_label_2[-3:]))
            self.pause()

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
            Z_COLOR = YELLOW
            # Make the Bell curve

            off_x = (max_bin_index + 0.5 + 0.25) / bins_per_unit

            sanity_plot = ax.plot(
                lambda x: np.exp(-((x - off_x) ** 2)),
                x_range=[0, n],
                color=HIST_COLOR,
            )

            graph = ax.plot(
                lambda x: (1 + 0.07 * ((x - off_x) ** 3 - 3 * (x - off_x)))
                * np.exp(-((x - off_x) ** 2) / (2 * 4 / 45 * n)),
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
            self.play(Create(graph))  # FadeIn(area),
            self.pause()

            # Create and animate the dashed line with label
            CLT_label_1 = MathTex(
                r"\text{Central Limit Theorem:}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            CLT_label_2 = MathTex(
                r"{{ X^2_{1} + \ldots + X^2_{n} }",
                r"\approx",
                r"\mathcal{N}",
                r"\bigg(",
                r"{n}",
                r"{1 \over 3}",
                r",",
                r"\sigma^2"
                r"\bigg)",
                # r"\! + \!",
                # r"\sqrt{ {n} }"
                # r"\frac{ 2 }{ 3 \sqrt{5} }",
                # r"Z",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            CLT_label_3 = MathTex(
                r"\text{1 Standard Deviation}",
                r"=",
                r"\sqrt{ {n} }"
                r"\frac{ 2 }{ 3 \sqrt{5} }",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            CLT_label_3[0].color = YELLOW

            CLT_label = (
                VGroup(CLT_label_1, CLT_label_2, CLT_label_3)
                .arrange(DOWN, buff=0.1)
                .next_to(LLN_eqn, DOWN)
                .to_edge(LEFT, buff=0.1)
            )

            self.play(Write(CLT_label_1))
            self.play(Write(CLT_label_2))
            self.pause()

            self.play(Write(CLT_label_3))
            self.pause()

            # Code to add the additional x-axis labels and underbrace
            offset = 0.3  # Adjust this as necessary for positioning the labels

            shift_amount = (
                np.sqrt(n) * 2 / 3 / np.sqrt(5)
            )  # Adjust this value to control the horizontal distance of the braces

            my_pos = n_over_3_label.get_bottom() + 0.1 * UP
            sqrt_n_term = r"\sqrt{ {n} } \frac{ 2 }{ 3 \sqrt{5} }"
            # Compute new positions for braces using a horizontal shift from the n/3 position
            right_brace_end = my_pos + np.array([shift_amount, 0, 0])
            left_brace_start = my_pos + np.array([-shift_amount, 0, 0])

            # Create the two braces with adjusted positions
            brace_color = YELLOW
            brace_dir = DOWN

            brace_right = BraceBetweenPoints(
                my_pos + 0.04 * RIGHT,
                right_brace_end,
                brace_dir,
                color=brace_color,
            )
            brace_left = BraceBetweenPoints(
                left_brace_start,
                my_pos + 0.04 * LEFT,
                brace_dir,
                color=brace_color,
            )

            # Create text labels using MathTex with tex_to_color_map
            my_fs = 28
            brace_text_right = MathTex(
                r"+1 \text{ SD}",
                tex_to_color_map=t2cD,
                font_size=my_fs,
                color=brace_color,
            )
            brace_text_left = MathTex(
                r"-1 \text{ SD}",
                tex_to_color_map=t2cD,
                font_size=my_fs,
                color=brace_color,
            )

            # Position the text labels at the center of the braces
            brace_text_right.next_to(brace_right, brace_dir, buff=0.1)
            brace_text_left.next_to(brace_left, brace_dir, buff=0.1)

            self.play(Create(brace_right), Write(brace_text_right))
            self.play(Create(brace_left), Write(brace_text_left))

            # Resuming the existing construct flow
            self.pause()

            self.next_section()

            location = bar_chart.bars[0 : 1 * bins_per_unit].get_center()

            P_le_1 = MathTex(
                r"\mathbb{P}\big( X^2_{1} + \ldots + X^2_{n} \le 1 \big)",
                tex_to_color_map=t2cD,
                font_size=42,
            ).move_to(location + 1.5 * DOWN)
            start_position = (
                P_le_1.get_top() + 0.2 * DOWN
            )  # Start just above the mobject

            # Define the end position, 'location' is the given end point for the arrow.
            # Ensure that the y-coordinate of 'location' is above the start_position's y-coordinate for an upward arrow
            end_position = location + 0.02 * DOWN

            # Create the arrow
            arrow = Arrow(
                start=start_position,
                end=end_position,
                color=GRAY_C,  # You can specify any color you prefer
            )
            self.play(Write(P_le_1))
            self.play(Write(arrow))
            self.pause()

            zoom_target = (
                bar_chart.bars[0 : 1 * bins_per_unit + 1].get_center()
                + 0.1 * DOWN
                + 0.05 * LEFT
            )
            self.zoomed_camera.frame.move_to(zoom_target)

            # Activate zooming and then animate the zoom-in effect after the histogram animation is done
            self.activate_zooming(animate=True)
            self.pause()

            large_deviations = MathTex(
                r"""\text{``Large deviation"}""",
                tex_to_color_map=t2cD,
                font_size=46,
            ).to_corner(UL, buff=0.65)
            large_deviations.set_z_index(
                1
            )  # large_deviations = large_deviations.set_z_index(1)
            self.play(Write(large_deviations))
            self.pause()

            self.wait(1)
            print("Done!")
    return (
        N_bins,
        N_sims,
        ProbabilitySpheres,
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
def _(ProbabilitySpheres, mo, render_scene):
    _video_path = render_scene(ProbabilitySpheres())
    mo.video(_video_path, loop=True, controls=True, autoplay=True)
    return


@app.cell(hide_code=True)
def _(
    BLUE,
    DOWN,
    FadeIn,
    FadeOut,
    GRAY_B,
    GRAY_C,
    GREEN,
    LEFT,
    MathTex,
    MoveToTarget,
    PI,
    RED,
    RIGHT,
    Scene,
    Text,
    Transform,
    TransformMatchingTex,
    UL,
    UP,
    UR,
    VGroup,
    Write,
    YELLOW_A,
    np,
):
    class ProbabilityDerivation(Scene):
        def pause(self):
            self.wait(0.1)
            # self.next_slide()

        def construct(self):
            self.next_section(skip_animations=True)
            non_var_color = GRAY_B
            n_color = BLUE
            lambda_color = RED
            t2cD = {
                r"\int": GRAY_B,
                r"\mathbf{1}": YELLOW_A,
                r"\mathbb{P}": non_var_color,
                r"\mathbb{E}": non_var_color,
                r"\bigg(": non_var_color,
                r"\bigg)": non_var_color,
                r"\bigg)^": non_var_color,
                r"\Big]": non_var_color,
                r"\Big[": non_var_color,
                r"\Big)": non_var_color,
                r"\Big(": non_var_color,
                r"\{": GRAY_C,
                r"\}": GRAY_C,
                r"{n}": n_color,
                r"{ n }": n_color,
                r"{  n  }": n_color,
                r"{\lambda}": lambda_color,
                r"{ \lambda }": lambda_color,
                r"X^2": GREEN,
                r"(X_1^2+\dots+X^2_{n})": GREEN,
                r"{X_1^2+\dots+X^2_{n}}": GREEN,
            }
            eqn_size = 41
            explanation_kwargs = {"font_size": 31}
            kwargs = {"font_size": eqn_size, "tex_to_color_map": t2cD}
            num_steps = 12

            lines = [None] * num_steps
            key_maps = [None] * (num_steps - 1)
            new_line = [None] * (num_steps - 1)
            explanation = [None] * (num_steps)

            # --- ANIMATION EXECUTION ---
            lhs = MathTex(
                r"\mathbb{P}",
                r"\Big(",
                r"{X_1^2+\dots+X^2_{n}}",
                r"{\le}",
                r"1",
                r"\Big)",
                **kwargs,
            )

            lhs_keymap = {
                r"\mathbb{P}": r"\mathbb{E}",
                r"\Big(": r"\Big[",
                r"\Big)": r"\Big]",
            }

            # --- STEP 0 ---
            lines[0] = MathTex(
                r"=",
                r"\mathbb{E}",
                r"\Big[",
                r"\mathbf{1}",
                r"_{",
                r"\{",
                r"{X_1^2+\dots+X^2_{n}}",
                r"{\le}",
                r"1",
                r"\}",
                r"}",
                r"\Big]",
                **kwargs,
            )
            # Transition 0 -> 1
            key_maps[0] = {r"=": r"\le"}
            new_line[0] = False

            # --- STEP 1 ---
            print(0)
            lines[1] = MathTex(
                r"\le",
                r"\mathbb{E}",
                r"\Big[",
                r"\mathbf{1}",
                r"_{",
                r"\{",
                r"{X_1^2+\dots+X^2_{n}}",
                r"{\le}",
                r"1",
                r"\}",
                r"}",
                r"{",
                r"{e}^{\lambda}",
                r"\over",
                r"e^{{\lambda}(X_1^2+\dots+X^2_{n})}",
                r"}",
                r"\Big]",
                **kwargs,
            )
            # Transition 1 -> 2: Removing the indicator and its subscript
            key_maps[1] = {}
            new_line[1] = False

            # --- STEP 2 ---
            lines[2] = MathTex(
                r"\le",
                r"\mathbb{E}",
                r"\Big[",
                r"{",
                r"{e}^{\lambda}",
                r"\over",
                r"e^{{\lambda}(X_1^2+\dots+X^2_{n})}",
                r"}",
                r"\Big]",
                **kwargs,
            )
            # Transition 2 -> 3: Fraction collapses into exponential with negative power
            key_maps[2] = {r"\le": r"=", r"e^{": r"e^{-"}
            new_line[2] = True
            # explanation[2] = VGroup(
            #    MathTex(r"(\text{``Exponential }", **explanation_kwargs),
            # MathTex(r"\text{Chebyshev Inequality''})", **explanation_kwargs),
            #       ).arrange(DOWN)

            explanation[2] = MathTex(
                r"\text{``Exponential Chebyshev Inequality''}", font_size=28
            )

            # --- STEP 3 ---
            lines[3] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\mathbb{E}",
                r"\Big[",
                r"e^{-{\lambda}(X_1^2+\dots+X^2_{n})}",
                r"\Big]",
                **kwargs,
            )
            # Transition 3 -> 4: Sum in exponent becomes i.i.d. product (raised to n)
            key_maps[3] = {
                r"(X_1^2+\dots+X^2_{n})": r"X^2",
            }
            new_line[3] = False

            # --- STEP 4 ---
            lines[4] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"\mathbb{E}",
                r"\Big[",
                r"e^{-",
                r"{\lambda}",
                r"X^2",
                r"}",
                r"\Big]",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 4 -> 5: E[] becomes an integral
            key_maps[4] = {
                r"\mathbb{E}": r"\int",
                r"X^2": r"x^2",
                # dx is new, so it will fade in via transform_mismatches=False
            }
            new_line[4] = False

            # --- STEP 5 ---
            lines[5] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"\int",
                r"^1",
                r"_{-1}",
                r"e^{-",
                r"{\lambda}",
                r"x^2",
                r"}",
                r"{",
                r"dx",
                r"\over",
                r"2",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 5 -> 6: Substitution/Scaling
            key_maps[5] = {
                r"^1": r"^{\sqrt{2",
                r"_{-1}": r"_{-\sqrt{2",
                r"{{\lambda} }": r"\frac{1}{2}",
                r"x^2": r"z^2",
                r"dx": r"dz",
            }
            #    r"^1": r"^{\sqrt{2{\lambda}}}",
            #    r"-{\lambda}": r"-\frac{1}{2}",
            #    r"x^2}": r"z^2}",
            #    r"dx": r"dz",
            # }
            new_line[5] = True
            explanation[5] = MathTex(
                r"\text{Def'n of i.i.d. } X\sim\text{Unif}[-1,1]",
                **explanation_kwargs,
            )

            ####
            lines[6] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg("
                r"\int",
                r"^{\sqrt{2",
                r"{ \lambda }",
                r"}}",
                r"_{-\sqrt{2",
                r"{ \lambda }",
                r"}}",
                r"e^{-",
                r"\frac{1}{2}",
                r"z^2",
                r"}",
                r"{",
                r"dz",
                r"\over",
                r"2",
                r"\sqrt",
                r"{",
                r"{{ 2 }}",
                r"{\lambda}}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )

            # Transition 6 -> 7: Changing limit of integration
            key_maps[6] = {
                r"=": r"\le",
                r"^{\sqrt{2": r"^{\infty}",
                r"_{-\sqrt{2": r"_{-\infty}",
            }
            new_line[6] = False

            # --- STEP 7 ---
            lines[7] = MathTex(
                r"\le",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"\int",
                r"^{\infty}",
                r"_{-\infty}",
                r"e^{-",
                r"\frac{1}{2}",
                r"z^2",
                r"}",
                r"{",
                r"dz",
                r"\over",
                r"2",
                r"\sqrt",
                r"{",
                r"{{ 2 }}",
                r"{\lambda}}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 7 -> 8: Symmetry (0 to inf becomes 1/2 of -inf to inf)

            # Transition 8 -> 9: Integral evaluated to sqrt(2pi)
            key_maps[7] = {
                r"\le": r"=",
                r"\int": r"\sqrt{",
                r"e^{-": r"{2}",
                r"dz": r"\pi}",
                r"2": r"{\frac{1}{2}}",
            }
            new_line[7] = True
            explanation[7] = MathTex(
                r"\text{Integrate to} \pm \infty", **explanation_kwargs
            )

            # --- STEP 9 ---
            lines[8] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"{\frac{1}{2}}",
                r"{",
                r"\sqrt{",
                r"{2}",
                r"\pi}",
                r"\over",
                r"\sqrt",
                r"{",
                r"{{ 2 }}",
                r"{\lambda}}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )

            # Transition 9 -> 10: Simplifying the sqrt terms
            key_maps[8] = dict()
            #   r"\sqrt{2{\lambda}}": r"\sqrt{{\lambda}}",
            #    r"\sqrt{2\pi}": r"\sqrt{\pi}",  # Absorbed
            # }
            new_line[8] = False
            explanation[8] = MathTex(
                r"\text{Evaluate }} \int_{-\infty}^\infty e^{-\frac{1}{2} z^2} dz =\sqrt{2\pi}",
                **explanation_kwargs,
            )

            # --- STEP 10 ---
            lines[9] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"{\frac{1}{2}}",
                r"{",
                r"\sqrt{",
                r"\pi}",
                r"\over",
                r"\sqrt",
                r"{",
                r"{\lambda}}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 10 -> 11: Plugging in optimal lambda = {n}/2
            key_maps[9] = {
                r"=": r"\le",
                r"{\lambda}": r"{ n }",
                r"{\lambda}": r"{  n  }",
            }
            new_line[9] = True

            # --- STEP 11 --- Plug in lambda = {n}/2
            lines[10] = MathTex(
                r"\le",
                r"{e}^{{ n }/2}",
                r"\bigg(",
                r"{\frac{1}{2}}",
                r"{",
                r"\sqrt{",
                r"\pi}",
                r"\over",
                r"\sqrt",
                r"{",
                r"{  n  }",
                r"/2}",
                r"}",
                r"\bigg)^",
                r"{n}",
                **kwargs,
            )
            new_line[10] = False

            key_maps[10] = {
                # Base e with power {n}/2 moves into the second fraction
                r"{e}^{": r"{ e }",
                r"{\frac{1}{2}}": r"2",
                r"\sqrt": r"/2}",
                r"\pi}": r"\pi",
                # The outer brackets change size/type
            }
            explanation[10] = MathTex(
                r"\text{Plug in optimal } {\lambda} = {n}/2",
                tex_to_color_map=t2cD,
                **explanation_kwargs,
            )

            # --- STEP 12 ---
            lines[11] = MathTex(
                r"\le",
                r"\bigg(",
                r"{",
                r"{ e }",
                r"\pi",
                r"\over",
                r"2",
                r"{  n  }",
                r"}",
                r"\bigg)",
                r"^{",
                r"{n}",
                r"/2}",
                **kwargs,
            )

            lhs.to_corner(UL, buff=0.3)
            lines[0].next_to(lhs, RIGHT, buff=0.2)
            lines[0].shift(0.06 * UP)

            self.play(Write(lhs))
            self.pause()

            eqn_run_time = 1

            self.play(
                TransformMatchingTex(
                    lhs.copy(),
                    lines[0],
                    key_map=lhs_keymap,
                    path_arc=PI / 2,
                    run_time=eqn_run_time,
                )
            )
            self.wait()

            all_on_screen = VGroup(lines[0])  # lhs,
            all_explanations = VGroup()
            current_rhs = lines[0]

            index_mob = Text("0").to_corner(UR)
            self.add(index_mob)

            explanation_x = 2.2

            for i in range(len(lines) - 1):
                next_index = Text(f"{i + 1}").to_corner(UR)
                next_line = lines[i + 1]
                if new_line[i]:  # True
                    next_line.next_to(
                        current_rhs, DOWN, aligned_edge=LEFT, buff=0.3
                    )
                    # if next_line.get_bottom()[1] < -3.5:
                    #    self.play(
                    #        all_on_screen.animate.shift(
                    #            UP * (next_line.get_height())
                    #        )
                    #    )
                    #    next_line.next_to(
                    #        current_rhs, DOWN, aligned_edge=LEFT, buff=0.4
                    #    )

                    self.play(
                        TransformMatchingTex(
                            current_rhs.copy(),
                            next_line,
                            key_map=key_maps[i],
                            transform_mismatches=False,
                            fade_transform_mismatches=False,
                        ),
                        Transform(index_mob, next_index),
                        run_time=eqn_run_time,
                    )

                    all_on_screen.add(next_line)
                    current_rhs = next_line
                else:
                    next_line.move_to(current_rhs, aligned_edge=LEFT)
                    self.play(
                        TransformMatchingTex(
                            current_rhs,
                            next_line,
                            key_map=key_maps[i],
                            transform_mismatches=False,
                            fade_transform_mismatches=False,
                            path_arc=PI / 2,
                        ),
                        Transform(index_mob, next_index),
                        run_time=eqn_run_time,
                    )
                    self.pause()
                    all_on_screen.remove(current_rhs)
                    all_on_screen.add(next_line)
                    current_rhs = next_line

                if explanation[i + 1]:
                    explanation[i + 1].next_to(current_rhs, RIGHT)
                    explanation[i + 1].set_x(explanation_x, LEFT)
                    self.play(Write(explanation[i + 1]))
                    self.pause()
                    all_explanations.add(explanation[i + 1])
            # { \pi^{ {n} \over 2} }
            ans = MathTex(
                r"{",
                r"\pi^",
                r"{n}",
                r"/2",
                r"\over",
                r"{",
                r"2^{ {n} }",
                r"( { n } / 2 )!",
                r"}",
                r"}",
                tex_to_color_map=t2cD,
                font_size=eqn_size,
            )

            print(*enumerate(ans), sep="\n")
            eq = MathTex(r"=", tex_to_color_map=t2cD, font_size=eqn_size)
            lhs.generate_target()
            current_rhs.generate_target()
            my_scale = 1.2
            VGroup(ans, eq, lhs.target, current_rhs.target).arrange(
                RIGHT, buff=0.2
            ).move_to(np.array([0, 0, 0])).scale(my_scale)
            # current_rhs.target.next_to(lhs, RIGHT, buff=0.2)
            self.next_section()
            self.play(
                MoveToTarget(lhs),
                MoveToTarget(current_rhs),
                *[
                    FadeOut(mob)
                    for mob in all_on_screen + all_explanations
                    if mob is not current_rhs
                ],
            )
            self.pause()
            self.play(FadeIn(eq, shift=LEFT))
            self.play(Write(ans))
            self.pause()

            ans.generate_target()
            VGroup(ans.target, current_rhs.target).arrange(
                RIGHT, buff=0.2
            ).move_to(np.array([0, 0, 0]))

            self.play(
                MoveToTarget(ans),
                MoveToTarget(current_rhs),
                FadeOut(eq),
                FadeOut(lhs),
            )
            self.pause()

            final_rhs = (
                MathTex(
                    r"\le",
                    r"\bigg(",
                    r"{",
                    r"{ e }",
                    r"\over",
                    r"{  n  }  / 2",
                    r"}",
                    r"\bigg)",
                    r"^{",
                    r"{n}",
                    r"/2}",
                    **kwargs,
                )
                .scale(my_scale)
                .align_to(current_rhs, LEFT)
            )

            final_lhs = (
                MathTex(
                    r"{",
                    r"1",
                    r"\over",
                    r"{",
                    r"( { n } / 2 )!",
                    r"}",
                    r"}",
                    tex_to_color_map=t2cD,
                    font_size=eqn_size,
                )
                .scale(my_scale)
                .align_to(ans, RIGHT)
            )

            self.play(
                TransformMatchingTex(ans, final_lhs, key_map={r"\pi^": r"1"}),
                TransformMatchingTex(current_rhs, final_rhs),
            )
            self.pause()

            final_rhs2 = (
                MathTex(
                    r"\ge",
                    r"\bigg(",
                    r"{",
                    r"{  n  } / 2",
                    r"\over",
                    r"{ e }",
                    r"}",
                    r"\bigg)",
                    r"^{",
                    r"{n}",
                    r"/2}",
                    **kwargs,
                )
                .scale(my_scale)
                .align_to(final_rhs, LEFT)
            )

            final_lhs2 = (
                MathTex(
                    r"( { n } / 2 )!",
                    tex_to_color_map=t2cD,
                    font_size=eqn_size,
                )
                .scale(my_scale)
                .next_to(final_rhs[0], LEFT, buff=0.2)
            )

            self.play(
                TransformMatchingTex(
                    final_lhs, final_lhs2, key_map={r"\pi^": r"1"}
                ),
                TransformMatchingTex(
                    final_rhs, final_rhs2, key_map={r"\le": r"\ge"}
                ),
            )
            self.pause()

            self.wait(3)
    return (ProbabilityDerivation,)


@app.cell
def _():
    return


@app.cell
def _(ProbabilityDerivation, mo, render_scene):
    _video_path = render_scene(ProbabilityDerivation())
    mo.video(_video_path, loop=True, controls=True, autoplay=True)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
