#%%manim_slides -ql -v WARNING Spider --manim-slides controls=false
#%%manim -ql -v WARNING Spider
# Choose to do regular manim or manim-slides by choosing the correct jupyter magic. Must be first line.

## Global Variables and Stuff ##
#config.media_embed = True
DELAY = 0.1
from manim import *
import math
import numpy as np
from manim_slides import *

# latex preamble
texPre = TexTemplate()
texPre.add_to_preamble(r"""
    \usepackage{amsmath}
    \usepackage{amssymb}
    \newcommand{\E}{\mathbb{E}}
    \newcommand{\P}{\mathbb{P}}
    \newcommand{\vect}[1]{\mathbf{#1}}
""") 

# tex color dictionary
E_color = GREY_B
n_color = BLUE_C
T_color = ORANGE
F_color = PURPLE_A
t2cD = {
    r"\E": E_color,
    r"\mathbb{P}": E_color,
    r"\big(": E_color,
    r"\big)": E_color,
    "n ": n_color,
    "T ": T_color,
    "F_": F_color
}

# font sizes
my_fs = 75

#other colors

house_color = WHITE
dot_color = WHITE
edge_color = BLUE

# Draw the spider's body
def my_spider(eye_angle = 0.1, my_scale = 0.5):
    body = Circle(radius=0.7, color=WHITE, fill_opacity=1).shift(UP * 0.5)

    # Draw the spider's eyes
    eye1 = Circle(radius=0.25, color=BLACK, fill_opacity=1).shift(UP * 0.7 + LEFT * 0.3)
    eye2 = Circle(radius=0.25, color=BLACK, fill_opacity=1).shift(UP * 0.7 + RIGHT * 0.3)


    # Add eyeballs to the eyes

    #eye_angle = 0.1
    eye_x = np.cos(eye_angle)*0.1
    eye_y = np.sin(eye_angle)*0.1
    eyeball1 = Circle(radius=0.1, color=WHITE, fill_opacity=1).shift(UP * 0.7 + RIGHT*eye_x + UP*eye_y + LEFT * 0.3)
    eyeball2 = Circle(radius=0.1, color=WHITE, fill_opacity=1).shift(UP * 0.7 + RIGHT*eye_x + UP*eye_y + RIGHT * 0.3)

    # Draw the spider's legs
    legs = VGroup()

    # Define rotation and positions for each leg
    leg_positions = [
        (-1,-1.5, 1), (-1,-1.5, 0.2),  # Left front legs
        (-1,-1.5, -0.5), (-1,-1.5, -1.2), # Left back legs
        (1,1.5, 1), (1,1.5, 0.2),    # Right front legs
        (1,1.5, -0.5), (1,1.5, -1.2)  # Right back legs
    ]

    # Add legs with two segments to the group
    for rot, x, y in leg_positions:
        theta = rot*0.2 

        frac = 0.8
        joint = [(x*np.cos(theta)-y*np.sin(theta)) * frac, (x*np.sin(theta)+y*np.cos(theta)) * frac, 0]  # Position of the joint between segments

        my_origin = [x*0.4, y*0.4,0]
        
        leg_width = my_scale*10
        leg_segment1 = Line(start=my_origin, end=joint, color=WHITE, stroke_width=leg_width ,path_arc=rot*0.9)
        frac = 0.78
        joint = [(x*np.cos(theta)-y*np.sin(theta)) * frac, (x*np.sin(theta)+y*np.cos(theta)) * frac, 0]  # Position of the joint between segments

        leg_segment2 = Line(start=joint, end=[x, y, 0], color=WHITE, stroke_width=leg_width ,path_arc=rot*0.75)
        leg_segment1.shift(UP * 0.5)  # Align first segment with the body
        leg_segment2.shift(UP * 0.5)  # Align second segment with the body
        legs.add(leg_segment1, leg_segment2)

    # Add everything to the scene eyeball1, eyeball2,

    return VGroup( body, eye1, eye2, legs, eyeball1, eyeball2).scale(my_scale).set_z_index(2)

def get_angle(mobject1, mobject2):
    # Get direction vectors of the two Mobjects
    delta_x = mobject2.get_x() - mobject1.get_x()
    delta_y = mobject2.get_y() - mobject1.get_y()

    # Calculate the angle between the two vectors
    return math.atan2(delta_y, delta_x)
    
## Global Variables to Store Final Mobjects Passed Between Scenes ##
FINAL_MOBJECTS = None
FINAL_MOBJECTS_2 = None

class Spider(Slide): #Scene): # change to MyScene(Slide) for manim-slides
    def pause(self):
        self.wait(DELAY)
        self.next_slide() #comment in for manim-slides
    def construct(self):
        #self.next_section(skip_animations=True) #comment out for manim-slides
        self.pause()
        
        ###
        #self.next_section() #comment out for manim-slied
        
        def my_house(x=0,y=0,scale=0.075):
            new_house = Polygon([0, 0, 0],  # Bottom-left base
                                [2.2, 0, 0],  # Bottom-left door
                                [2.2, 2.8, 0],  # Top-left door
                                [3.8, 2.8, 0],  # Top-right door
                                [3.8, 0, 0],  # Bottom-right door
                                [6, 0, 0],  # Bottom-right base
                                [6, 4, 0],  # Top-right base
                                [6.5, 4, 0],   # Rightmost corner roof
                                [3, 7, 0],     # Apex roof
                                [-0.5, 4, 0],  # Leftmost corner roof
                                [0, 4, 0],  # Top-left base
                               color=WHITE, fill_color=WHITE, fill_opacity=1.0)
            new_house.set_x(x)
            new_house.set_y(y)
            new_house.scale(scale)
            return new_house
        
        
        
        pentagon = RegularPolygon(n=5).scale(2.0)
        pentagon.shift(0.1*DOWN) #to_edge(RIGHT,buff=1.0)
        #self.add(pentagon)
        # Get the vertices of the polygon
        vertices = pentagon.get_vertices()
        edges = VGroup(*[Line(vertices[i], vertices[(i + 1) % 5], color=edge_color) for i in range(5)])


        # Add the polygon to the scene
        

        # Mark the vertices
        dots = VGroup(*[Dot(point=vertex, color=dot_color, radius=0.1) for vertex in vertices])
       
        #house_size = 2.0 
        #house = VGroup( Polygon([house_size/2,0,0],[0,house_size/2,0],[-house_size/2,0,0])  )
        
        #house += Square(side_length=house[0].width*0.75).next_to(house[0],DOWN,buff=0.0)
        #self.add(house)
        
        #house.move_to(dots[0])

        # Optionally, add labels to the vertices
        for i, vertex in enumerate(vertices):
            label = Text(f"{i+1}").scale(0.5).next_to(vertex, UP * 0.2)
            #self.add(label)

        spider = my_spider(eye_angle = 90*DEGREES)
        
        spider.move_to(dots[0])

        #self.add(spider)
        
        
        #self.add(edges)
        #self.add(dots[1:])
        
        
        #home = MathTex(r"\text{HOME}",font_size=50)
        house = my_house()
        house.set_z_index(1)
        house.move_to(dots[0])
        #self.add(house)
        #home.next_to(spider,UP)
        #self.add(home)

        
        q_size = 60
        eq_size = 60
        hint_size = 40
        line1 = MathTex(r'\text{A spider goes for a random walk on a pentagon.}',font_size=q_size)
        line2 = MathTex( r'T', '=', r'\text{ The number of steps until return home.}',font_size=q_size)
        line2[0].set_color(T_color)

        probs_list = MathTex(r'{ 1 ', r' \over 2 } ', '+',
                            r'{ 0 ', r' \over 4 }', '+',
                            r' { 1 ', r' \over 8 } ', '+',
                            r'{ 1 ', r' \over 16 } ', '+',
                            r'{ 2 ', r' \over 32 } ', '+',
                            r'{ 3 ', r' \over 64 } ', '+',
                            r'{ 5 ', r' \over 128 } ', '+',
                            r'{ 8 ', r' \over 256 } ', r' \ldots ',
                        font_size=1.5*hint_size, tex_to_color_map=t2cD)

        probs_list2 = MathTex(r'{ 1 ', r' \over 2 } ', ',',
                            r'{ 0 ', r' \over 4 }', ',',
                            r' { 1 ', r' \over 8 } ', ',',
                            r'{ 1 ', r' \over 16 } ', ',',
                            r'{ 2 ', r' \over 32 } ', ',',
                            r'{ 3 ', r' \over 64 } ', ',',
                            r'{ 5 ', r' \over 128 } ', ',',
                            r'{ 8 ', r' \over 256 } ', r' \ldots ',
                        font_size=1.5*hint_size, tex_to_color_map=t2cD)

        probs_list.set_color_by_tex('+',BLACK)
        eq_one = MathTex('=','1',font_size=1.5*hint_size, tex_to_color_map=t2cD )
        
       

        P_exact_hint = MathTex(r'\text{Exactly } n \text{ steps:}',
                        font_size=hint_size, tex_to_color_map=t2cD)
        P_exact = MathTex(r'\mathbb{P}\big(T = n \big) = { F_{ n - 3}', r' \over 2^{ n -1 } }',
                        font_size=eq_size, tex_to_color_map=t2cD) #,
                        #tex_template=texPre) #, tex_to_color_map=t2cD)
        P_g_hint = MathTex(r'\text{Still out after } n \text{ steps:}',
                        font_size=hint_size, tex_to_color_map=t2cD)
        P_g = MathTex(r'\mathbb{P}\big(T > n \big) = { F_{ n }', r' \over 2^{ n -1 } }',
                        font_size=eq_size, tex_to_color_map=t2cD)
        
        top_buff = 0.3
        line1.to_edge(UP,buff=top_buff)
        line2.to_edge(UP,buff=top_buff)
        #line2.align_to(line1,LEFT)
        
        P_exact.set_y(spider.get_y())
        P_exact.to_edge(LEFT)
        
        P_g.set_y(spider.get_y())
        P_g.to_edge(RIGHT)
        
        P_exact_hint.next_to(P_exact,UP,buff=0.05).align_to(P_exact,LEFT)
        P_g_hint.next_to(P_g,UP,buff=0.05).align_to(P_g,LEFT)

        self.play(LaggedStart( [FadeIn(probs_list[i],shift=RIGHT) for i in range(len(probs_list))],lag_ratio=0.1))
        self.pause()
        


        probs_list.generate_target()
        probs_list.target.set_color_by_tex("+", WHITE)
        my_group = VGroup(probs_list.target, eq_one).arrange(RIGHT)
        self.play(MoveToTarget(probs_list),FadeIn(eq_one,shift=RIGHT))
        self.pause()
        
        #probs_list.generate_target()
        #probs_list.target.set_color_by_tex("+", BLACK)
        probs_list2.next_to(pentagon,DOWN,buff=0.7)
        self.play(TransformMatchingTex(probs_list,probs_list2),FadeOut(eq_one))
        self.pause()
        
        self.play(LaggedStart( [Write(line1),FadeIn(spider),FadeIn(house),FadeIn(dots[1:]), FadeIn(edges)] ,lag_ratio=0.1))
        self.pause()

        #self.play(FadeIn(edges),FadeIn(dots[1:]),FadeIn(house))
        #self.pause()

        #return 0
        
        #self.play(Write(line1))
        
        
        spider_pos = 0
        spider_pos_post = 0
        for t in range(8):
            choice = np.random.choice([-1,+1])
            #left_step = ( spider_pos - 1 ) % 5
            #right_step = ( spider_pos + 1 ) % 5
            #spider_pos_post = np.random.choice([left_step,right_step])
            #while spider_pos_post == 0:
            #    spider_pos_post = ( spider_pos + np.random.choice([-1,1]) ) % 5
            
            
            
            print(spider_pos_post)
            
            look_length = 1.0
            look_num = 2
            for spider_target in [-1,+1]*look_num:
                spider_pos_post = ( spider_pos + choice*spider_target ) % 5
                spider.generate_target()
                spider.target = my_spider(eye_angle = get_angle(dots[spider_pos],dots[spider_pos_post]))

                spider.target.move_to(dots[spider_pos]) #move just the eyes!
                self.play(MoveToTarget(spider),run_time=look_length/(2.0*look_num))
                
            spider_pos_post = ( spider_pos + choice ) % 5
                
            spider.target.move_to(dots[spider_pos_post])
            anim_group = [MoveToTarget(spider)]
            if t==5:
                anim_group.append(LaggedStart(FadeOut(line1,shift=RIGHT),Write(line2),lag_ratio=0.5))
                #anim_group.append()

            #if t==8:
            #    anim_group.append(ReplacementTransform(probs_list2.copy(), VGroup(P_exact, P_exact_hint)))

            #if t==9:
            #    anim_group.append(ReplacementTransform(probs_list2.copy(), VGroup(P_g, P_g_hint)))
    
            self.play(anim_group)
            if t in [5]: #,8,9]:
                self.pause()
            

            spider_pos = spider_pos_post

        self.play(ReplacementTransform(probs_list2.copy(), VGroup(P_exact, P_exact_hint)))
        self.pause()

        self.play(ReplacementTransform(probs_list2.copy(), VGroup(P_g, P_g_hint)))
        self.pause()
        
        self.play(FadeOut(line2,P_exact,P_exact_hint,P_g,P_g_hint, probs_list2))
        self.pause()
                
        global FINAL_MOBJECTS, FINAL_MOBJECTS_2
        vs = VGroup(house, dots[1], dots[2], dots[3],dots[4], house.copy())
        FINAL_MOBJECTS = (spider, edges, vs)
        FINAL_MOBJECTS_2 = spider_pos
    
#%%manim -ql -v WARNING Unfold
# %%manim_slides -ql -v WARNING MyScene --manim-slides controls=true
# Choose to do regular manim or manim-slides by choosing the correct jupyter magic. Must be first line.

def mob_arrow(start_mob: Mobject, end_mob: Mobject, **kwargs) -> Arrow:
    """
    Create an arrow from the right edge of start_mob to the left edge of end_mob.

    Additional Arrow arguments can be passed via kwargs.
    """
    return Arrow(
        start=start_mob.get_edge_center(RIGHT),
        end=end_mob.get_edge_center(LEFT),
        color = GREY_C,
        stroke_width=1.5,
        tip_length = 0.15,
        **kwargs
    )


def background_box(width=5, height=None, font_size=24, box_opacity=0.8):
    """
    Create a paragraph inside a black background box.

    Args:
        text (str): The text content.
        width (float): Width of the box.
        height (float or None): Height of the box. If None, adjusts automatically.
        font_size (int): Size of the text.
        box_opacity (float): Opacity of the black background.

    Returns:
        VGroup: A group containing the background box and the text.
    """
    # Create the paragraph
    #paragraph = Paragraph(text, alignment="center", font_size=font_size)
    #paragraph.width = width

    # Optional: shrink text to fit if it gets too wide
    #if paragraph.width > width:
    #    paragraph.scale_to_fit_width(width)

    # Create a background box
    rect_height = height if height else paragraph.height + 0.5
    background = Rectangle(
        width=width + 0.5,  # padding
        height=rect_height,
        fill_color=BLACK,
        fill_opacity=box_opacity,
        stroke_color=WHITE,
        stroke_width=1
    )

    # Layer: background behind paragraph
    #group = VGroup(background, paragraph)
    #group.arrange(DOWN, center=True, aligned_edge=UP)

    return background


class Unfold(Slide): #Scene): # change to MyScene(Slide) for manim-slides
    def pause(self):
        self.wait(DELAY)
        self.next_slide() #comment in for manim-slides
    def construct(self):
    
        #self.next_section() #skip_animations=True) #comment out for manim-slides
        #self.pause()
        
        ###
        #self.next_section() #comment out for manim-slied
        spider, edges, vs = [mob.copy() for mob in FINAL_MOBJECTS]
        spider_pos = FINAL_MOBJECTS_2
        #for i in range(5):
        #    self.play(Create(vs[i]))
        #    self.play(Create(edges[i]))
        #self.play(FadeOut(vs[0]))
        #self.play(Create(vs[5]))    
        #return 0
        self.add(spider, edges, vs)
        
        spacing = vs[3].get_x() - vs[2].get_x() #distance between dots
        
        new_y = 0 #vs[2].get_y()
        
        unfolded_vs = VGroup(*[vertex.copy() for vertex in vs] )
        for i in range(5+1):
            unfolded_vs[i].set_x(spacing*(i-2) + vs[2].get_x())
            unfolded_vs[i].set_y(new_y)
        
        #unfolded_edges = VGroup(*[Line([spacing*(i-2) + vs[2].get_x(),new_y,0],
        #                               [spacing*(i-1) + vs[2].get_x(),new_y,0],
        #                                color=edge_color) for i in range(5)])
        
        #self.add(unfolded_edges)
        
        #dumb way to do it because im getting frustrated looking up how to make the partial plug ins work correctly
        def update_line_0(mob):
            mob.put_start_and_end_on(vs[0].get_center(),vs[1].get_center())
        def update_line_1(mob):
            mob.put_start_and_end_on(vs[1].get_center(),vs[2].get_center())
        def update_line_2(mob):
            mob.put_start_and_end_on(vs[2].get_center(),vs[3].get_center())
        def update_line_3(mob):
            mob.put_start_and_end_on(vs[3].get_center(),vs[4].get_center())
        def update_line_4(mob):
            mob.put_start_and_end_on(vs[4].get_center(),vs[5].get_center())
        def update_to_spider_pos(mob):
            mob.move_to(vs[spider_pos])
            
        edges[0].add_updater(update_line_0)
        edges[1].add_updater(update_line_1)
        edges[2].add_updater(update_line_2)
        edges[3].add_updater(update_line_3)
        edges[4].add_updater(update_line_4)
        spider.add_updater(update_to_spider_pos)
        
        for i in range(5+1):
            vs[i].generate_target()
            vs[i].target = unfolded_vs[i]
        
        #spider.generate_target()
        #spider.target.move_to(unfolded_vs[spider_pos])
        
        #self.play(vs[0].animate.shift(LEFT))
        
        arc_angles = [90*DEGREES,72*DEGREES,0*DEGREES,-0*DEGREES,-72*DEGREES,-90*DEGREES]
        self.play(*[MoveToTarget(vs[i],path_arc=arc_angles[i]) for i in range(5+1)])
        self.pause()

        #spider_pos = 0
        spider_pos_post = 0
        for t in range(6):
            choice = np.random.choice([-1,+1])
            if spider_pos == 5:
                choice = -1
            if spider_pos == 0:
                choice = 1
            #left_step = ( spider_pos - 1 ) % 5
            #right_step = ( spider_pos + 1 ) % 5
            #spider_pos_post = np.random.choice([left_step,right_step])
            #while spider_pos_post == 0:
            #    spider_pos_post = ( spider_pos + np.random.choice([-1,1]) ) % 5
            
            
            
            print(spider_pos_post)
            
            look_length = 1.0
            look_num = 2
            for spider_target in [-1,+1]*look_num:
                spider_pos_post = ( spider_pos + choice*spider_target ) % 6
                spider.generate_target()
                spider.target = my_spider(eye_angle = get_angle(vs[spider_pos],vs[spider_pos_post]))

                spider.target.move_to(vs[spider_pos]) #move just the eyes!
                self.play(MoveToTarget(spider),run_time=look_length/(2.0*look_num))
                
            spider_pos_post = ( spider_pos + choice ) % 6
                
            spider.target.move_to(vs[spider_pos_post])
            anim_group = [MoveToTarget(spider)]
            self.play(anim_group)

            spider_pos = spider_pos_post

        #return 0


        #,
        #         MoveToTarget(spider,path_arc=arc_angles[spider_pos]))
        #self.play(
        #          MoveToTarget(vs[1],path_arc=72*DEGREES), #,
        #          MoveToTarget(vs[2]),
        #          MoveToTarget(vs[3]),
        #          MoveToTarget(vs[4],path_arc=-72*DEGREES),
        #          MoveToTarget(vs[0],path_arc=90*DEGREES),
        #          MoveToTarget(spider,path_arc=90*DEGREES),
        #          MoveToTarget(vs[5],path_arc=-90*DEGREES),
        #         )
        
        rotated_vs = VGroup(*[vertex.copy() for vertex in vs] )
        delta_x = 1.1
        delta_y = 0.9 #(2*frame_height2 - top_buff - bottom_buff) / 5
        
        frame_height2 = config.frame_height/2
        frame_width2 = config.frame_width/2
        #top_buff = delta_y/2.0 #0.5
        #bottom_buff = 2.5 #distance from bottom
        for i in range(5+1):
            rotated_vs[i].set_y(frame_height2 - delta_y*(i+0.5))
            rotated_vs[i].set_x(delta_x*(0.5) - frame_width2)
        
        my_scale = 0.65
        for i in range(5+1):
            vs[i].generate_target()
            vs[i].target = rotated_vs[i]
            vs[i].target.scale(my_scale)
        

        self.pause()
        self.play(*[MoveToTarget(vs[i]) for i in range(5+1)],spider.animate.scale(my_scale))
        self.pause()

        spider.clear_updaters()
        spider.generate_target()        
        spider.target.move_to(rotated_vs[5])
        self.play(MoveToTarget(spider,path_arc=-90*DEGREES))
        self.pause()
        




        
        gridline_color = GREY
        x_final = -frame_width2 + delta_x*(9+1)
        gridlines = VGroup(*[Line( [-frame_width2,frame_height2 - delta_y*(i+1),0],
                                   [x_final,frame_height2 - delta_y*(i+1),0],
                                  color=gridline_color, stroke_width=2) for i in range(6)])

        #gridline_color = GREY
        y_final = frame_height2 - delta_y*(5+1)
        gridlines_x = VGroup(*[Line( [-frame_width2 + delta_x*(i+1), frame_height2 , 0],
                                   [-frame_width2 + delta_x*(i+1), y_final , 0],
                                  color=gridline_color, stroke_width=1) for i in range(9+1)])
                          
        
        #print(delta_y)
        
        time_color = GREY 
        time_fs = 50
        max_t = 9
        times = VGroup(*[MathTex(f"{t}",font_size=time_fs,color=time_color) for t in range(max_t+1)])
        times.add(MathTex(r"n ",font_size=time_fs,tex_to_color_map=t2cD))
        #times[-1][-1].color = BLACK #make the 1 invisible! r"\!+\!1",
        times.add(MathTex(r"n \!+\!1",font_size=time_fs,tex_to_color_map=t2cD))
        
        
        
        for t in range(max_t+3):
            times[t].next_to(gridlines[-1],DOWN)
            times[t].set_x(delta_x*(t+0.5) - frame_width2)
        last_x_shift = 0.5 #amount to shift over the very last dude.
        times[-1].shift(last_x_shift*RIGHT)
        
        times[-1].align_to(times[0],DOWN) #reset the height of this 
        
        times[-2].align_to(times[-1][0],DOWN) #alignt the "n"s
        
        time_label = MathTex(r"n \text{ steps}",font_size=time_fs,color=time_color,tex_to_color_map=t2cD)
        
        time_label.next_to(gridlines[-1],DOWN)
        time_label.to_edge(RIGHT)
        
        self.play(Create(gridlines))
        self.play(Create(gridlines_x))
        self.pause()
        
        self.play(Write(time_label))
        self.play(Create(times[1:max_t+1])) #[1:max_t+1]))
        self.pause()


        N_ways = VGroup(MathTex(r"\# \text{ of ways to }"), MathTex(r"\text{get to square}")).arrange(DOWN)
        N_ways.to_corner(UR,buff=0.25)
        #N_ways.next_to(vs[0],RIGHT, buff = 0.7)
        self.play(Write(N_ways))
        self.pause()


        
        #self.next_section()
        
        
        
        
        grid_nums = [[None for y in range(5+1)] for t in range(max_t+2+1)]
        
        my_nums = [[1,1,1], #format: t,y, value
                   [2,0,1],
                   [2,2,1],
                   [3,1,1],
                   [3,3,1],
                   [4,0,1],
                   [4,2,2],
                   [4,4,1],
                   [5,1,2],
                   [5,3,3],
                   [5,5,1],
                   [6,0,2],
                   [6,2,5],
                   [6,4,3],
                   [7,1,5],
                   [7,3,8],
                   [7,5,3],
                   [8,0,5],
                   [8,2,13],
                   [8,4,8],
                   [9,1,13],
                   [9,3,21],
                   [9,5,8],
                   ]
        grid_fs = 60
        grid_color = WHITE
        for t,y,val in my_nums:
            grid_nums[t][y] = MathTex(f"{val}",font_size=grid_fs,color=grid_color)
        

        #assume that max_t is odd
        fgrid_fs = 0.8*grid_fs
        grid_nums[max_t+1][0] = MathTex(r"F_{n -3}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+1][2] = MathTex(r"F_{n -1}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+1][4] = MathTex(r"F_{n -2}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+2][1] = MathTex(r"F_{n -1}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+2][3] = MathTex(r"F_{n }",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+2][5] = MathTex(r"F_{n -2}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        
        
        
        old_arrows = VGroup()
        new_arrows = VGroup()

        for t in range(1,max_t+3):
            anim_group = []
            old_arrows += new_arrows
            new_arrows = VGroup()
            if t == max_t + 1:

                x_final = frame_width2
                gridlines2 = VGroup(*[Line( [-frame_width2,frame_height2 - delta_y*(i+1),0],
                                        [x_final,frame_height2 - delta_y*(i+1),0],
                                        color=gridline_color, stroke_width=2) for i in range(6)])

                #gridline_color = GREY
                y_final = frame_height2 - delta_y*(5+1)
                gridlines_x2 = VGroup(*[Line( [-frame_width2 + delta_x*(i+1), frame_height2 , 0],
                                        [-frame_width2 + delta_x*(i+1), y_final , 0],
                                        color=gridline_color, stroke_width=1) for i in range(9,11)])
                            
                self.play(Create(gridlines2),Create(gridlines_x2),FadeOut(N_ways),FadeOut(time_label))
                self.play(LaggedStart(FadeIn(times[-2]),FadeIn(times[-1]) ,lag_ratio=0.1))
                self.pause()
                #anim_group.append(ReplacementTransform(times[t-1].copy(),grid_nums[max_t+1][0]))
            for y in range(5+1):
                if grid_nums[t][y]:
                    grid_nums[t][y].set_x(delta_x*(t+0.5) - frame_width2)
                    grid_nums[t][y].set_y(frame_height2 - delta_y*(5-y+0.5))
                    if t==max_t+2:
                        grid_nums[t][y].shift(last_x_shift*RIGHT) #amount to shift over the very last dude.
        
                    #self.add(grid_nums[t][y])
                    if grid_nums[t][y]:
                        ancestors = VGroup()
                        if t ==1 :
                            ancestors += spider
                            new_arrows += mob_arrow(spider[0],grid_nums[t][y])
                        if t >= 2:
                            if y >= 2 and grid_nums[t-1][y-1]:
                                ancestors += grid_nums[t-1][y-1]
                                new_arrows += mob_arrow(grid_nums[t-1][y-1],grid_nums[t][y])
                            if y < 4 and grid_nums[t-1][y+1]:
                                ancestors += grid_nums[t-1][y+1]
                                new_arrows += mob_arrow(grid_nums[t-1][y+1],grid_nums[t][y])
                            
                    #self.play(Create(grid_nums[t][y]))
                        anim_group.append(ReplacementTransform(ancestors.copy(),grid_nums[t][y]))
                        #self.play(FadeIn(new_arrows))

                        #anim_group.append(FadeIn(new_arrows))

            self.play(*anim_group, *[Write(a) for a in new_arrows]) #, FadeOut(old_arrows))
            self.pause()

        for i in [1,3,5]:
            grid_nums[max_t+2][i].generate_target()
        
        grid_nums[max_t+2][1].target = MathTex(r"F_{n \!+\!1\!-\!2}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+2][3].target = MathTex(r"F_{n \!+\!1\!-\!1}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        grid_nums[max_t+2][5].target = MathTex(r"F_{n \!+\!1\!-\!3}",font_size=fgrid_fs,color=grid_color,tex_to_color_map=t2cD)
        
        anim_list = []
        for i in [1,3,5]:
            grid_nums[max_t+2][i].target.move_to(grid_nums[max_t+2][i])
            grid_nums[max_t+2][i].target.align_to(grid_nums[max_t+2][i],LEFT)
            anim_list.append(MoveToTarget(grid_nums[max_t+2][i]))
            #anim_list.append(TransformMatchingTex(grid_nums[max_t+2][i],grid_nums[max_t+2][i].target))

        self.play(*anim_list)
        self.pause()

        #self.next_section()

        
        claim_fs = 48
        claim_statement = MathTex(r"\text{ Claim:}", r"\text{ Number of ways at time }", r"n ",r"\text{ are: }", font_size=claim_fs, tex_to_color_map=t2cD)
        claim_fib = MathTex(r"F_{n -3}, F_{n -1}, F_{n -2}",font_size = fgrid_fs, tex_to_color_map=t2cD)
        claim_proof = MathTex(r"\text{ Proof:}", r"\text{ By induction! }",font_size=claim_fs)
        claim_text = VGroup(claim_statement,claim_fib,claim_proof).arrange(DOWN)
        claim_proof.align_to(claim_statement,LEFT)
        claim_text += Underline(claim_statement[0])
        proof_u = Underline(claim_proof[0])
        claim_text += proof_u
        claim_box = background_box(width = claim_text.width, height = claim_text.height + 0.5, font_size = 24, box_opacity = 0.9)
        claim_text.set_z_index(4)
        claim_box.set_z_index(3)
        claim = VGroup(claim_text,claim_box)
        claim.to_corner(UL,buff=0.25)
        claim_text.remove(claim_proof)
        claim_text.remove(proof_u)
        self.play(FadeIn(claim,shift=DOWN))
        self.pause()

        self.play(FadeIn(VGroup(claim_proof,proof_u),shift=DOWN))
        self.pause()
        
        #corr_statement = MathTex(r"\text{ Corrolary:}", r"\text{ Number of ways at time }", r"n ",r"\text{ are: }", font_size=claim_fs, tex_to_color_map=t2cD)
        #P_exact_hint = MathTex(r'\text{Exactly } n \text{ steps:}',
        #               font_size=hint_size, tex_to_color_map=t2cD)
        P_corr_exact = MathTex(r"\text{ Corollary: }", r'\mathbb{P}\big(T = n \big) = { F_{ n - 3}', r' \over 2^{ n -1 } }',
                        font_size=claim_fs, tex_to_color_map=t2cD) #,
        P_corr_exact.set_z_index(4)
                        #tex_template=texPre) #, tex_to_color_map=t2cD)
        #P_g_hint = MathTex(r'\text{Still out after } n \text{ steps:}',
        #                font_size=hint_size, tex_to_color_map=t2cD)
        #P_g = MathTex(r'\mathbb{P}\big(T > n \big) = { F_{ n }', r' \over 2^{ n -1 } }',
        #                font_size=eq_size, tex_to_color_map=t2cD)

        corr_box = background_box(width = claim_text.width, height = P_corr_exact.height + 0.2, font_size = 24, box_opacity = 0.9)
        corr = VGroup(P_corr_exact,corr_box)

        buff_y = 0.15
        corr.next_to(claim,DOWN,buff=buff_y)
        P_corr_exact[0].align_to(claim_proof[0],LEFT)
        corr_u = Underline(P_corr_exact[0])
        corr += corr_u

        self.play(FadeIn(corr,shift=DOWN))
        self.pause()

        #self.next_section()
        
        #corr_statement = MathTex(r"\text{ Corrolary:}", r"\text{ Number of ways at time }", r"n ",r"\text{ are: }", font_size=claim_fs, tex_to_color_map=t2cD)
        #P_exact_hint = MathTex(r'\text{Exactly } n \text{ steps:}',
        #               font_size=hint_size, tex_to_color_map=t2cD)
        P_corr_g = MathTex(r"\text{ Corollary: }", r'\mathbb{P}\big(T > n \big) = { F_{ n - 1 } + F_{ n - 2 } ', r' \over 2^{ n -1 } }',
                        font_size=claim_fs, tex_to_color_map=t2cD) #,

        P_corr_g.set_z_index(4)
                        #tex_template=texPre) #, tex_to_color_map=t2cD)
        #P_g_hint = MathTex(r'\text{Still out after } n \text{ steps:}',
        #                font_size=hint_size, tex_to_color_map=t2cD)
        #P_g = MathTex(r'\mathbb{P}\big(T > n \big) = { F_{ n }', r' \over 2^{ n -1 } }',
        #                font_size=eq_size, tex_to_color_map=t2cD)

        corr_box2 = background_box(width = claim_text.width, height = P_corr_g.height + 0.2, font_size = 24, box_opacity = 0.9)
        corr_box2.set_z_index(3)
        corr2 = VGroup(P_corr_g,corr_box2)

        corr2.next_to(corr,DOWN,buff=buff_y)
        P_corr_g[0].align_to(claim_proof[0],LEFT)
        corr_u2 = Underline(P_corr_g[0]).set_z_index(3)
        corr2 += corr_u2




        self.play(FadeIn(corr2,shift=DOWN))
        self.pause()


        P_corr_g_FINAL = MathTex(r"\text{ Corollary: }", r'\mathbb{P}\big(T > n \big) = { F_{ n }', r' \over 2^{ n -1 } }',
                        font_size=claim_fs, tex_to_color_map=t2cD) #,

        P_corr_g_FINAL.set_z_index(4)
        P_corr_g_FINAL.move_to(P_corr_g)
        P_corr_g_FINAL[0].align_to(P_corr_g[0],LEFT)

        self.play(TransformMatchingTex(P_corr_g,P_corr_g_FINAL))
        self.pause()

        #self.remove(time_label)
        #self.add(times[-2],times[-1]) #add the n and n+1
        #self.pause()
        self.wait(2)


        
                  
        
        
  
# %%
