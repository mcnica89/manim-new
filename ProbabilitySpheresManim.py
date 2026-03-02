import marimo

__generated_with = "0.12.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from manim_slides import Slide
    import manim
    import numpy as np


    # This manually injects everything from manim into the local namespace
    # essentially does from manim impoort *, but in a way that marimo allows it
    for name in dir(manim):
        if not name.startswith("_"):
            globals()[name] = getattr(manim, name)
    return Slide, manim, mo, name, np


@app.cell
def _(
    Axes,
    BLUE,
    BarChart,
    Create,
    DOWN,
    DashedLine,
    FunctionGraph,
    GREEN,
    LEFT,
    MathTex,
    PURPLE,
    RED,
    RIGHT,
    Slide,
    Transform,
    UL,
    UP,
    VGroup,
    Write,
    YELLOW,
    ZoomedScene,
    np,
    truncated_normal_distribution,
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
    t2cD = {"R": RED, "n": n_color, "{1}": n_color, "{2}": n_color, "X": GREEN}

    my_zoom_scale_factor = 4


    class ProbabilitySpheres(ZoomedScene, Slide):
        def __init__(self, **kwargs):
            ZoomedScene.__init__(
                self,
                zoomed_display_corner=np.array([-1, 1, 0.0]),
                zoom_factor=0.3,
                zoomed_display_height=16 / my_zoom_scale_factor,
                zoomed_display_width=16 / my_zoom_scale_factor,
                zoomed_camera_config={
                    "default_frame_stroke_width": 3,
                },
                **kwargs,
            )

        def pause(self):
            self.next_slide()

        def construct(self):
            print("Constructing...")
            self.wait(0.1)
            self.pause()
            self.wait(0.1)
            self.next_section(skip_animations=True)

            def my_hist_values(_samples):
                hist_values = np.zeros(N_bins - 1)
                for sample in _samples:
                    bin_index = np.digitize(sample, bins) - 1
                    bin_index = min(bin_index, len(hist_values) - 1)
                    hist_values[bin_index] += 1
                return hist_values

            complete_hist_values = my_hist_values(samples)
            max_hist_value = np.max(complete_hist_values)

            def my_bar_chart(_hist_values):
                low_color = RED
                high_color = BLUE
                chart = BarChart(
                    values=_hist_values,
                    y_range=[0, max_hist_value, max_hist_value],
                    bar_colors=[
                        low_color if i < bins_per_unit else high_color
                        for i in range(N_bins - 1)
                    ],
                    y_length=4.5,
                    x_length=8,
                    x_axis_config={"font_size": 24},
                    y_axis_config={"font_size": 1},
                )
                chart.to_edge(RIGHT)
                chart.to_edge(UP, buff=1.0)
                return chart

            hist_values = np.zeros(N_bins - 1)
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
                r"n", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[-1].get_center() + DOWN * down_len)
            label_mid = MathTex(
                r"{n}/2", tex_to_color_map=t2cD, font_size=label_fs
            ).move_to(bar_chart.bars[N_bins // 2].get_center() + DOWN * down_len)
            label = MathTex(
                r"\text{Histogram of } R^2 = X^2_{1} + X^2_{2} + \ldots + X^2_n",
                tex_to_color_map=t2cD,
            ).next_to(bar_chart, UP)

            sample_fs = 36
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
                    MathTex(r"\text{Samples:} ", str(m), font_size=24)
                    .next_to(label, DOWN)
                    .align_to(label, RIGHT)
                )

            self.next_section()

            samples_label = my_samples_label(0)
            self.play(Write(samples_label))
            self.pause()

            def my_new_label(m):
                return (
                    MathTex(r"\text{Samples:} ", "{:,}".format(m), font_size=24)
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
                r"n/3", tex_to_color_map=t2cD, font_size=label_fs
            ).next_to(dashed_line, DOWN, buff=0)
            n_over_3_label.shift(DOWN * down_len)

            # Create and animate the dashed line with label
            LLN_label_1 = MathTex(
                r"\text{Law of Large Numbers:}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            LLN_label_2 = MathTex(
                r"{R^2 \over",
                r"n",
                r"}",
                r"=",
                r"{{ X^2_{1} + \ldots + X^2_n }",
                r"\over",
                r"n",
                r"}",
                r"\to",
                r"{1 \over 3}",
                tex_to_color_map=t2cD,
                font_size=label_fs,
            )
            LLN_label = (
                VGroup(LLN_label_1, LLN_label_2).arrange(DOWN).to_corner(UL)
            )

            self.play(
                Create(dashed_line), Write(n_over_3_label), Write(LLN_label)
            )  # Create and animate the dashed line with label

            std_dev = 0.1

            # Define the parameters
            HIST_WIDTH = 8
            HIST_HEIGHT = 4.5
            HIST_COLOR = HIST_LEFT_COLOR = RED
            HIST_RIGHT_COLOR = BLUE
            bell_curve_max_x = n / 3  # Center it at the expected value

            # Create new axes
            ax = Axes(
                x_range=[0, n, n / 2],
                y_range=[0, max_hist_value, max_hist_value / 2],
                tips=False,
                x_length=HIST_WIDTH,
                y_length=HIST_HEIGHT,
                axis_config={"include_numbers": False},
            )

            # Define the normal distribution really modified to match histogram
            def normal_distribution(x, mean, std_dev, height):
                norm_factor = height / (std_dev * np.sqrt(2 * np.pi))
                exponent = -((x - mean) ** 2) / (2 * std_dev**2)
                return norm_factor * np.exp(exponent)

            # Plot the normal distribution
            graph = ax.plot(
                lambda x: normal_distribution(
                    x, bell_curve_max_x, std_dev, max_hist_value
                ),
                x_range=[0, n],
                color=PURPLE,
                use_smoothing=True,
            )

            # Overlap the axes, bars, and bell curve
            # chart = my_bar_chart(hist_values)
            # chart.move_to(ax.c2p(0, 0))
            ax.align_to(bar_chart.y_axis, LEFT).align_to(bar_chart.x_axis, DOWN)
            graph.align_to(bar_chart.y_axis, LEFT).align_to(bar_chart.x_axis, DOWN)
            self.play(Create(graph))

            self.wait(2)

            return 0
            # Calculate the mean position on the x-axis in terms of the histogram's axis
            x_axis = bar_chart.x_axis
            mean_position = x_axis.number_to_point(max_bin_index + 0.5)[
                0
            ]  # Centered at the dotted line

            # Ensure the height of the normal distribution matches with y-axis
            y_axis_scale_factor = (
                y_axis.number_to_point(1)[1] - y_axis.number_to_point(0)[1]
            )  # Scale factor for y-axis
            normal_curve = FunctionGraph(
                lambda x: truncated_normal_distribution(
                    x, mean_position, std_dev, max_hist_value * y_axis_scale_factor
                ),
                x_range=[0, n],  # Truncated between 0 and n
                color=PURPLE,
            )

            # Position and transform the graph to align with the histogram's axes
            # normal_curve.stretch_to_fit_height(max_hist_value * y_axis_scale_factor)
            # normal_curve.move_to(np.array_axis.number_to_point(0))
            normal_curve.align_to(bar_chart.y_axis, DOWN).align_to(
                bar_chart.x_axis, LEFT
            )

            # Add the normal distribution to the scene
            self.play(Create(normal_curve), run_time=2)
            self.pause()

            return 0
            self.next_section()

            zoom_target = (
                bar_chart.bars[0 : 1 * bins_per_unit].get_center() + 0.1 * DOWN
            )
            self.zoomed_camera.frame.move_to(zoom_target)

            # Activate zooming and then animate the zoom-in effect after the histogram animation is done
            self.activate_zooming(animate=True)
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
        samples,
        t2cD,
    )


@app.cell
def _():
    return


@app.cell
def _(
    BLUE,
    DOWN,
    GRAY_B,
    GRAY_C,
    GREEN,
    LEFT,
    MathTex,
    RED,
    RIGHT,
    ReplacementTransform,
    Scene,
    Text,
    Transform,
    TransformMatchingTex,
    UL,
    UP,
    UR,
    VGroup,
    Write,
    YELLOW_E,
):
    class ProbabilityDerivation(Scene):
        def construct(self):
            non_var_color = GRAY_C
            t2cD = {
                r"\int": GRAY_B,
                r"\mathbf{1}": YELLOW_E,
                r"\mathbb{E}": non_var_color,
                r"\bigg(": non_var_color,
                r"\bigg)": non_var_color,
                r"\bigg)^": non_var_color,
                r"\Big]": non_var_color,
                r"\Big[": non_var_color,
                r"\{": GRAY_B,
                r"\}": GRAY_B,
                r"{n}": BLUE,
                r"\lambda": RED,
                r"(X_1^2+\dots+X^2_n)": GREEN,
                r"{X_1^2+\dots+X^2_n}": GREEN,
            }
            kwargs = {"font_size": 40, "tex_to_color_map": t2cD}
            num_steps = 13

            lines = [None] * num_steps
            key_maps = [None] * (num_steps - 1)
            new_line = [None] * (num_steps - 1)

            # --- STEP 0 ---
            lines[0] = MathTex(
                r"=",
                r"\mathbb{E}",
                r"\Big[",
                r"\mathbf{1}",
                r"_{\{ {X_1^2+\dots+X^2_n} \le 1\}}",
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
                r"_{\{ {X_1^2+\dots+X^2_n} \le 1\}}",
                r"{",
                r"{e}^{\lambda}",
                r"\over",
                r"e^{\lambda(X_1^2+\dots+X^2_n)}",
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
                r"e^{\lambda(X_1^2+\dots+X^2_n)}",
                r"}",
                r"\Big]",
                **kwargs,
            )
            # Transition 2 -> 3: Fraction collapses into exponential with negative power
            key_maps[2] = {r"\le": r"=", r"e^{": r"e^{-"}
            new_line[2] = True

            # --- STEP 3 ---
            lines[3] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\mathbb{E}",
                r"\Big[",
                r"e^{-\lambda(X_1^2+\dots+X^2_n)}",
                r"\Big]",
                **kwargs,
            )
            # Transition 3 -> 4: Sum in exponent becomes i.i.d. product (raised to n)
            key_maps[3] = {
                r"(X_1^2+\dots+X^2_n)": r"X^2",
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
                r"\lambda",
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
                r"\bigg( ",
                r"\int",
                r"^1",
                r"_0",
                r"e^{-",
                r"\lambda",
                r"x^2",
                r"}",
                r"dx",
                r"\bigg)^{n}",
                **kwargs,
            )
            print(*enumerate(lines[5]), sep="\n")
            # Transition 5 -> 6: Substitution/Scaling
            key_maps[5] = {
                r"^1": r"^{\sqrt{2",
                r"{\lambda }": r"\frac{1}{2}",
                r"x^2": r"z^2",
                r"dx": r"{dz \over \sqrt{2\lambda}}",
            }
            #    r"^1": r"^{\sqrt{2\lambda}}",
            #    r"-\lambda": r"-\frac{1}{2}",
            #    r"x^2}": r"z^2}",
            #    r"dx": r"dz",
            # }
            new_line[5] = False

            # --- STEP 6 ---
            lines_6_temp = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg( ",
                r"\int",
                r"^{\sqrt{2\lambda}}",
                r"_0",
                r"e^{-",
                r"\frac{1}{2}",
                r"z^2",
                r"}",
                r"{dz \over \sqrt{2\lambda}}",
                r"\bigg)^{n}",
                **kwargs,
            )

            ####
            lines[6] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg("
                r"\int",
                r"^{\sqrt{2\lambda}}",
                r"_0",
                r"e^{-",
                r"\frac{1}{2}",
                r"z^2",
                r"}",
                r"{",
                r"dz",
                r"\over",
                r"\sqrt",
                r"{",
                r"2",
                r"\lambda}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )

            print(*enumerate(lines[6]), sep="\n")
            # Transition 6 -> 7: Changing limit of integration
            key_maps[6] = {r"=": r"\le", r"^{\sqrt{2\lambda}}": r"^{\infty}"}
            new_line[6] = False

            # --- STEP 7 ---
            lines[7] = MathTex(
                r"\le",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"\int",
                r"^{\infty}",
                r"_0",
                r"e^{-",
                r"\frac{1}{2}",
                r"z^2",
                r"}",
                r"{",
                r"dz",
                r"\over",
                r"\sqrt",
                r"{",
                r"2",
                r"\lambda}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 7 -> 8: Symmetry (0 to inf becomes 1/2 of -inf to inf)
            key_maps[7] = {r"\le": r"=", r"_0": r"_{-\infty}"}
            new_line[7] = False

            # --- STEP 8 ---
            lines[8] = MathTex(
                r"=",
                r"{e}^{\lambda}",
                r"\bigg(",
                r"{\frac{1}{2}}",
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
                r"\sqrt",
                r"{",
                r"2",
                r"\lambda}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 8 -> 9: Integral evaluated to sqrt(2pi)
            key_maps[8] = {
                r"\int": r"\sqrt{",
                r"e^{-": r"{2}",
                r"dz": r"\pi}",
            }
            new_line[8] = False

            # --- STEP 9 ---
            lines[9] = MathTex(
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
                r"2",
                r"\lambda}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 9 -> 10: Simplifying the sqrt terms
            key_maps[9] = dict()
            #   r"\sqrt{2\lambda}": r"\sqrt{\lambda}",
            #    r"\sqrt{2\pi}": r"\sqrt{\pi}",  # Absorbed
            # }
            new_line[9] = False

            # --- STEP 10 ---
            lines[10] = MathTex(
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
                r"\lambda}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )
            # Transition 10 -> 11: Plugging in optimal lambda = {n}/2
            key_maps[11] = {
                r"e^{\lambda}": r"e^{{n}/2}",
                r"\lambda}": r"{n}/2}",
            }
            new_line[10] = True

            # --- STEP 11 --- Plug in lambda = {n}/2
            lines[11] = MathTex(
                r"=",
                r"{e}^{{n}/2}",
                r"\bigg(",
                r"{\frac{1}{2}}",
                r"{",
                r"\sqrt{",
                r"\pi}",
                r"\over",
                r"\sqrt",
                r"{",
                r"{n}/2}",
                r"}",
                r"\bigg)^{n}",
                **kwargs,
            )

            new_line[11] = True

            key_maps[11] = {
                # Base e with power {n}/2 moves into the second fraction
                r"{e}^": r" e",
                r"{\frac{1}{2}}": r"2",
                # The outer brackets change size/type
                r"\pi}": r"\pi",
            }

            # --- STEP 12 ---
            lines[12] = MathTex(
                r"=",
                r"\bigg(",
                r"{",
                r"\pi"
                r" e",
                r"\over",
                r"2",
                r"{n}",
                r"}",
                r"\bigg)^{{n}/2}",
                **kwargs,
            )

            # --- ANIMATION EXECUTION ---
            lhs = MathTex(r"\mathbb{P}(X_1^2+\dots+X^2_n \le 1)", **kwargs)
            lhs.to_corner(UL, buff=1)
            lines[0].next_to(lhs, RIGHT, buff=0.2)

            self.add(lhs)
            self.play(Write(lines[0]))
            self.wait()

            all_on_screen = VGroup(lhs, lines[0])
            current_rhs = lines[0]

            index_mob = Text("0").to_corner(UR)
            self.add(index_mob)

            for i in range(len(lines) - 1):
                next_index = Text(f"{i + 1}").to_corner(UR)
                next_line = lines[i + 1]
                if i == 50:
                    lines_6_temp.move_to(current_rhs, aligned_edge=LEFT)
                    self.play(ReplacementTransform(current_rhs, lines_6_temp))
                    lines[6].move_to(current_rhs, aligned_edge=LEFT)
                    current_rhs = lines[6]
                    self.remove(lines_6_temp)
                    self.add(current_rhs)
                else:
                    if new_line[i]:
                        next_line.next_to(
                            current_rhs, DOWN, aligned_edge=LEFT, buff=0.4
                        )
                        if next_line.get_bottom()[1] < -3.5:
                            self.play(all_on_screen.animate.shift(UP * 3.0))

                        self.play(
                            TransformMatchingTex(
                                current_rhs.copy(),
                                next_line,
                                key_map=key_maps[i],
                                transform_mismatches=False,
                                fade_transform_mismatches=False,
                            ),
                            Transform(index_mob, next_index),
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
                            ),
                            Transform(index_mob, next_index),
                        )
                        all_on_screen.remove(current_rhs)
                        all_on_screen.add(next_line)
                        current_rhs = next_line
                self.wait(0.5)

            self.wait(3)
    return (ProbabilityDerivation,)


@app.cell
def _():
    return


@app.cell
def _(ProbabilityDerivation, config, mo):
    config.quality = "low_quality"
    config.media_dir = "media"
    config.verbosity = "WARNING"
    config.progress_bar = "display"


    def render_scene():
        scene = ProbabilityDerivation()
        scene.render()

        # This dynamically retrieves the path to the movie file
        # that Manim just generated!
        return scene.renderer.file_writer.movie_file_path


    video_path = render_scene()
    mo.video(video_path, loop=True, controls=True, autoplay=True)
    return render_scene, video_path


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
