##########
import marimo as mo
from manim_slides import Slide
import manim

# This manually injects everything from manim into the local namespace
# essentially does from manim impoort *, but in a way that marimo allows it
for name in dir(manim):
    if not name.startswith("_"):
        globals()[name] = getattr(manim, name)

##########
class SCENE_NAME(Slide):  # lide):
    def pause(self):
        # self.wait(DELAY)
        self.next_slide()  # comment in for manim-slides
        # self.wait(0.1)
        # pass

    def construct(self):
        self.wait(0.1)
        self.pause()
        self.wait(0.1)
        self.next_section(skip_animations=True)



###########
config.quality = "high_quality"
config.media_dir = "media"


def render_scene():
    scene = SCENE_NAME()
    scene.render()

    # This dynamically retrieves the path to the movie file
    # that Manim just generated!
    return scene.renderer.file_writer.movie_file_path


video_path = render_scene()
mo.video(video_path, loop=True, controls=True, autoplay=True)