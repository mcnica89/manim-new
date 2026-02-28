from manim import *
from manim_slides import *
# Default Global Colors
V_DOMINO_COLOR  = BLUE_C
H_DOMINO_COLOR  = BLUE_D

# --- Highlight Colors (for the new tiles) ---
HIGHLIGHT_H_COLOR = RED_D
HIGHLIGHT_V_COLOR = RED_C

DELAY = 0.1
my_fs = 32
title_fs = 40

class DominoPatterns(Slide):  #lide):
    def pause(self):
        #self.wait(DELAY)
        self.next_slide() #comment in for manim-slides
        #pass
    def construct(self):
        self.next_section() 
        # 1. Set White Background
        self.camera.background_color = WHITE
        
        # 2. Title
        title = Text("What's the pattern?", color=BLACK, font_size=title_fs)
        title.to_edge(UP, buff=0.1)
        self.add(title)

        # --- Modular Subroutine for Drawing ---
        # UPDATED: Added default_h_color and default_v_color arguments
        def draw_tiling(horiz_coords, vert_coords, rows, cols, 
                        h_highlight=None, v_highlight=None,
                        default_h_color=H_DOMINO_COLOR, default_v_color=V_DOMINO_COLOR):
            tiling_group = VGroup()
            
            # Create the background grid
            grid = VGroup(*[
                Square(side_length=1).set_stroke(BLACK, 3) 
                for _ in range(rows * cols)
            ]).arrange_in_grid(rows=rows, cols=cols, buff=0)
            tiling_group.add(grid)

            # Helper to get center of a grid cell
            def get_cell_pos(r, c):
                index = r * cols + c
                return grid[index].get_center()

            # Shared style for dominos
            domino_kwargs = {
                "corner_radius": 0.1,
                "fill_opacity": 1.0, 
                "stroke_width": 2
            }

            # Draw Horizontal Dominos
            for d in horiz_coords:
                p1, p2 = get_cell_pos(*d[0]), get_cell_pos(*d[1])
                
                # Check if this domino should be colored differently
                fill_col = default_h_color  # Use local default
                if h_highlight and d in h_highlight:
                    fill_col = HIGHLIGHT_H_COLOR

                dom = RoundedRectangle(height=0.8, width=1.8, **domino_kwargs)
                dom.set_fill(fill_col) 
                dom.move_to((p1 + p2) / 2)
                tiling_group.add(dom)

            # Draw Vertical Dominos
            for d in vert_coords:
                p1, p2 = get_cell_pos(*d[0]), get_cell_pos(*d[1])

                # Check if this domino should be colored differently
                fill_col = default_v_color # Use local default
                if v_highlight and d in v_highlight:
                    fill_col = HIGHLIGHT_V_COLOR

                dom = RoundedRectangle(height=1.8, width=0.8, **domino_kwargs)
                dom.set_fill(fill_col) 
                dom.move_to((p1 + p2) / 2)
                tiling_group.add(dom)

            return tiling_group.scale(1.5)

        # --- Modular Subroutine for Column Layout ---
        # UPDATED: Added h_color and v_color arguments
        def create_column(top_label, bottom_label, items, x_position, rows, cols, scale_val=0.5, 
                          h_color=H_DOMINO_COLOR, v_color=V_DOMINO_COLOR):
            """
            top_label:   Mobject for "k Dominos"
            bottom_label: Mobject for "n Ways"
            """
            # Anchor the top label
            top_label.move_to([x_position, 3.0 + top_label.get_height()/2, 0])
            
            shapes = VGroup()
            
            for item in items:
                if isinstance(item, dict):
                    # Extract optional highlight lists
                    hl_h = item.get('h_highlight', [])
                    hl_v = item.get('v_highlight', [])
                    
                    # Pass custom colors down to drawing function
                    t = draw_tiling(item['h'], item['v'], rows, cols, 
                                    h_highlight=hl_h, v_highlight=hl_v,
                                    default_h_color=h_color, default_v_color=v_color)
                    shapes.add(t)
                elif isinstance(item, Mobject):
                    shapes.add(item)
            
            shapes.arrange(DOWN, buff=0.4)
            shapes.scale(scale_val)
            
            # Place shapes under the top label
            shapes.next_to(top_label, DOWN, buff=0.15)

            # Place bottom label under shapes
            bottom_label.next_to(shapes, DOWN, buff=0.15)

            # Center them all horizontally
            bottom_label.align_to(top_label, ORIGIN)
            shapes.align_to(top_label, ORIGIN)

            return VGroup(top_label, shapes, bottom_label)

        # --- DATA DEFINITIONS ---

        # Chunk 1: 1 Domino (1 way)
        chunk1_data = [{'h': [], 'v': [[[0,0],[1,0]]]}]

        # Chunk 2: 2 Dominos (2 ways)
        chunk2_data = [
            {'h': [], 'v': [[[0,0],[1,0]], [[0,1],[1,1]]]},
            {'h': [[[0,0],[0,1]], [[1,0],[1,1]]], 'v': []}
        ]

        # Chunk 3: 3 Dominos (3 ways)
        chunk3_data = [
            {'h': [], 'v': [[[0,0],[1,0]], [[0,1],[1,1]], [[0,2],[1,2]]]},
            {'h': [[[0,0],[0,1]], [[1,0],[1,1]]], 'v': [[[0,2],[1,2]]]},
            {'h': [[[0,1],[0,2]], [[1,1],[1,2]]], 'v': [[[0,0],[1,0]]]}
        ]

        # Chunk 4: 4 Dominos (5 ways)
        chunk4_data = [
            {'h': [], 'v': [[[0,0],[1,0]], [[0,1],[1,1]], [[0,2],[1,2]], [[0,3],[1,3]]]},
            {'h': [[[0,0],[0,1]], [[1,0],[1,1]]], 'v': [[[0,2],[1,2]], [[0,3],[1,3]]]},
            {'h': [[[0,1],[0,2]], [[1,1],[1,2]]], 'v': [[[0,0],[1,0]], [[0,3],[1,3]]]},
            {'h': [[[0,2],[0,3]], [[1,2],[1,3]]], 'v': [[[0,0],[1,0]], [[0,1],[1,1]]]},
            {'h': [[[0,0],[0,1]], [[1,0],[1,1]], [[0,2],[0,3]], [[1,2],[1,3]]], 'v': []}
        ]

        # Chunk 5: 5 Dominos (Unknown ways)
        my_scale = 3
        dots = MathTex(r"\vdots", color=BLACK).scale(my_scale)
        qmark = MathTex(r"???", color=BLACK).scale(my_scale)
        chunk5_items = [
            {'h': [], 'v': []},
            dots.copy(),
            qmark.copy(),
            dots.copy(),
            {'h': [], 'v': []}
        ]

        # --- RENDER ---
        
        x_positions = [-7 + 14/5 * (i+0.5) for i in range(5)]

        def make_top_label(text):
            return Text(text, color=BLACK, font_size=my_fs)

        def make_bottom_label(text):
            return Text(text, color=BLACK, font_size=my_fs)

        w = 1.4

        col = [None] * 6
        col_0_v = MAROON_C
        col_0_h = MAROON_D
        col[0] = (create_column(
            make_top_label("1 domino"),
            make_bottom_label("1 way"),
            chunk1_data, x_positions[0], rows=2, cols=1, scale_val=w/2,
            v_color=col_0_v, h_color=col_0_h
        ))

        col_1_v = GREEN_C 
        col_1_h = GREEN_D
        col[1] = (create_column(
            make_top_label("2 dominos"),
            make_bottom_label("2 ways"),
            chunk2_data, x_positions[1], rows=2, cols=2, scale_val=w/2.5,
            v_color=col_1_v, h_color=col_1_h
        ))

        # Example: Set Col 3 to GREEN
        col_3_v = BLUE_C
        col_3_h = BLUE_D
        col[2] = (create_column(
            make_top_label("3 dominos"),
            make_bottom_label("3 ways"),
            chunk3_data, x_positions[2], rows=2, cols=3, scale_val=w/3,
            v_color=col_3_v, h_color=col_3_h
        ))

        # Example: Set Col 4 to TEAL
        col_4_v = PURPLE_C
        col_4_h = PURPLE_D
        col[3] = (create_column(
            make_top_label("4 dominos"),
            make_bottom_label("5 ways"),
            chunk4_data, x_positions[3], rows=2, cols=4, scale_val=w/4,
            v_color=col_4_v, h_color=col_4_h
        ))

        col[4] = (create_column(
            make_top_label("5 dominos"),
            make_bottom_label("??? ways"),
            chunk5_items.copy(), x_positions[4], rows=2, cols=5, scale_val=w/5
        ))

        self.add(*[col[i] for i in range(5)])
        self.wait(2)
        self.pause()

        #### ANIMATION 1

        new_num_cols = 6
        new_x_positions = [-7 + 14/new_num_cols * (i+0.5) for i in range(6)]

        self.play(*[col[i].animate.shift(RIGHT* (new_x_positions[i] - x_positions[i])) for i in range(5)], title.animate.shift(LEFT*2))
        
        col[4].generate_target()
        col[4].target = (create_column(
            make_top_label("5 dominos\n End Horiz."),
            make_bottom_label("??? ways"),
             [
            {'h': [], 'v': []},
            dots.copy(),
            qmark.copy(),
            dots.copy(),
            {'h': [], 'v': []}
        ], new_x_positions[4], rows=2, cols=5, scale_val=w/5
        ))

        col[5] = (create_column(
            make_top_label("5 dominos\n End Vert."),
            make_bottom_label("??? ways"),
            [
            {'h': [], 'v': []},
            dots.copy(),
            qmark.copy(),
            dots.copy(),
            {'h': [], 'v': []}
        ], new_x_positions[5], rows=2, cols=5, scale_val=w/5
        ))

        self.play(MoveToTarget(col[4]),ReplacementTransform(col[4].copy(), col[5]))
        
        self.pause()

        #### ANIMATE IN END BIT

        my_scale = 3
        dots = MathTex(r"\vdots", color=BLACK).scale(my_scale)
        qmark = MathTex(r"???", color=BLACK).scale(my_scale)
        chunk5_items_H = [
            {'h': [[[0,3],[0,4]],[[1,3],[1,4]]], 'v': [], 'h_highlight': [[[0,3],[0,4]],[[1,3],[1,4]]]},
            dots.copy(),
            qmark.copy(),
            dots.copy(),
             {'h': [[[0,3],[0,4]],[[1,3],[1,4]]], 'v': [], 'h_highlight': [[[0,3],[0,4]],[[1,3],[1,4]]]}
        ]

        chunk5_items_V = [
            {'h': [], 'v': [[[0,4],[1,4]]], 'v_highlight': [[[0,4],[1,4]]]},
            dots.copy(),
            qmark.copy(),
            dots.copy(),
            {'h': [], 'v': [[[0,4],[1,4]]], 'v_highlight': [[[0,4],[1,4]]]}
        ]

        col[4].generate_target()
        col[4].target = (create_column(
            make_top_label("5 dominos\n End Horiz."),
            make_bottom_label("??? ways"),
            chunk5_items_H, new_x_positions[4], rows=2, cols=5, scale_val=w/5
        ))

        col[5].generate_target()
        col[5].target = (create_column(
            make_top_label("5 dominos\n End Vert."),
            make_bottom_label("??? ways"),
            chunk5_items_V, new_x_positions[5], rows=2, cols=5, scale_val=w/5
        ))

        self.play(Transform(col[4],col[4].target), Transform(col[5],col[5].target))
        self.pause()

        ### replace ends in horiz now

        new_col4 = col[2].copy()
        new_col4[0] = col[4][0].copy()

        top_label,shapes,bottom_label = new_col4
        # Place shapes under the top label
        shapes.next_to(top_label, DOWN, buff=0.15)

        # Place bottom label under shapes
        bottom_label.next_to(shapes, DOWN, buff=0.15)

        # Center them all horizontally
        bottom_label.align_to(top_label, ORIGIN)
        shapes.align_to(top_label, ORIGIN)

        chunk5_HORIZ = [
            # 1. Originally 3 Verticals -> Now has 2 Horizontals (added first)
            {
                'h': [[[0,3],[0,4]], [[1,3],[1,4]]],
                'v': [[[0,0],[1,0]], [[0,1],[1,1]], [[0,2],[1,2]]],
                'h_highlight': [[[0,3],[0,4]], [[1,3],[1,4]]] 
            },

            # 2. Originally 2 Horiz (Start), 1 Vert -> Now 2 Horiz (End) + 2 Horiz (Start)
            {
                'h': [[[0,3],[0,4]], [[1,3],[1,4]], [[0,0],[0,1]], [[1,0],[1,1]]],
                'v': [[[0,2],[1,2]]],
                'h_highlight': [[[0,3],[0,4]], [[1,3],[1,4]]] 
            },

            # 3. Originally 1 Vert, 2 Horiz (End) -> Now 2 Horiz (End) + 2 Horiz (Mid)
            {
                'h': [[[0,3],[0,4]], [[1,3],[1,4]], [[0,1],[0,2]], [[1,1],[1,2]]],
                'v': [[[0,0],[1,0]]],
                'h_highlight': [[[0,3],[0,4]], [[1,3],[1,4]]] 
            }
        ]

        # Use Col 3 colors for the column derived from Col 3
        col_4new = (create_column(
            make_top_label("5 dominos\n End Horiz."),
            make_bottom_label("3 ways"),
            chunk5_HORIZ, new_x_positions[4], rows=2, cols=5, scale_val=w/5,
            v_color=col_3_v, h_color=col_3_h 
        ))
        self.play(ReplacementTransform(col[2][1:].copy(),col_4new[1:]),FadeOut(col[4][1:]))
        self.pause()

        ### replace ends in vertical now (col[5])

        # Copy the 4-domino column (col[3]) structure as a base
        new_col5 = col[3].copy()
        new_col5[0] = col[5][0].copy() # Copy label from destination

        top_label, shapes, bottom_label = new_col5
        
        # Place shapes under the top label
        shapes.next_to(top_label, DOWN, buff=0.15)

        # Place bottom label under shapes
        bottom_label.next_to(shapes, DOWN, buff=0.15)

        # Center them all horizontally
        bottom_label.align_to(top_label, ORIGIN)
        shapes.align_to(top_label, ORIGIN)

        # Define the 5 patterns ending in vertical
        chunk5_VERT = [
            # 1. Originally 4 Verticals -> Now New Vert + 4 Verts
            {
                'h': [], 
                'v': [[[0,4],[1,4]], [[0,0],[1,0]], [[0,1],[1,1]], [[0,2],[1,2]], [[0,3],[1,3]]],
                'v_highlight': [[[0,4],[1,4]]] 
            },
            # 2. Originally 2 Horiz (Start), 2 Vert -> Now New Vert + 2 Verts
            {
                'h': [[[0,0],[0,1]], [[1,0],[1,1]]], 
                'v': [[[0,4],[1,4]], [[0,2],[1,2]], [[0,3],[1,3]]],
                'v_highlight': [[[0,4],[1,4]]] 
            },
            # 3. Originally 2 Horiz (Mid), 2 Vert -> Now New Vert + 2 Verts
            {
                'h': [[[0,1],[0,2]], [[1,1],[1,2]]], 
                'v': [[[0,4],[1,4]], [[0,0],[1,0]], [[0,3],[1,3]]],
                'v_highlight': [[[0,4],[1,4]]] 
            },
            # 4. Originally 2 Horiz (End), 2 Vert -> Now New Vert + 2 Verts
            {
                'h': [[[0,2],[0,3]], [[1,2],[1,3]]], 
                'v': [[[0,4],[1,4]], [[0,0],[1,0]], [[0,1],[1,1]]],
                'v_highlight': [[[0,4],[1,4]]] 
            },
            # 5. Originally 4 Horiz (2 sets), 0 Vert -> Now New Vert
            {
                'h': [[[0,0],[0,1]], [[1,0],[1,1]], [[0,2],[0,3]], [[1,2],[1,3]]], 
                'v': [[[0,4],[1,4]]],
                'v_highlight': [[[0,4],[1,4]]] 
            }
        ]

        # Use Col 4 colors for the column derived from Col 4
        col_5new = (create_column(
            make_top_label("5 dominos\n End Vert."),
            make_bottom_label("5 ways"),
            chunk5_VERT, new_x_positions[5], rows=2, cols=5, scale_val=w/5,
            v_color=col_4_v, h_color=col_4_h
        ))

        # Transform col[3] (4-domino source) into col_5new, fading out the placeholder
        self.play(ReplacementTransform(col[3][1:].copy(), col_5new[1:]), FadeOut(col[5][1:]))
        self.pause()

        # --- Add Brace and Total Label ---
        
        brace = Brace(VGroup(col_4new[2], col_5new[2]), DOWN).set_color(BLACK)
        total_text = brace.get_text("Total: 8 ways").set_color(BLACK)
        self.play(GrowFromCenter(brace), Write(total_text))
        self.pause()