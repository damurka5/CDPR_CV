from mujoco.glfw import glfw
import mujoco as mj
import numpy as np

class CDPRViewer:
    def __init__(self, model, width=1200, height=900):
        self.model = model
        self.width = width
        self.height = height
        self._init_glfw()
        self._init_viewer()
        self._setup_callbacks()
        
        # Store actual framebuffer size
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(self.window)
        
        # Camera control state
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.last_x = 0
        self.last_y = 0

    def _init_glfw(self):
        glfw.init()
        self.window = glfw.create_window(self.width, self.height, "CDPR Simulation", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

    def _init_viewer(self):
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        
        # Set better initial camera parameters
        self.cam.azimuth = 45
        self.cam.elevation = -30
        self.cam.distance = 3.5
        self.cam.lookat = np.array([0.0, 0.0, 1.5])  # Focus on robot center
        
        # Initialize scene and context
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        
        # Set this flag for proper rendering
        self.context.offWidth = self.width
        self.context.offHeight = self.height
    
    def _setup_callbacks(self):
        glfw.set_key_callback(self.window, self._keyboard)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move)
        glfw.set_mouse_button_callback(self.window, self._mouse_button)
        glfw.set_scroll_callback(self.window, self._scroll)
        glfw.set_window_size_callback(self.window, self._window_resize)

    def _window_resize(self, window, width, height):
        self.width = width
        self.height = height
        self.fb_width, self.fb_height = glfw.get_framebuffer_size(window)
        self.context.offWidth = self.fb_width
        self.context.offHeight = self.fb_height
        
    def _keyboard(self, window, key, scancode, act, mods):
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            # Reset simulation
            pass

    def _mouse_button(self, window, button, act, mods):
        self.button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
        self.button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
        self.button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
        
        # Update mouse position
        glfw.get_cursor_pos(window)

    def _mouse_move(self, window, xpos, ypos):
        dx = xpos - self.last_x
        dy = ypos - self.last_y
        self.last_x = xpos
        self.last_y = ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = (glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS) or \
                    (glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS)

        # Determine action based on mouse button
        if self.button_right:
            action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
        elif self.button_left:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(self.model, action, dx/height, dy/height, self.scene, self.cam)

    def _scroll(self, window, xoffset, yoffset):
        action = mj.mjtMouse.mjMOUSE_ZOOM
        mj.mjv_moveCamera(self.model, action, 0.0, -0.05*yoffset, self.scene, self.cam)

    def update(self, model, data):
        # Get current window size
        width, height = glfw.get_framebuffer_size(self.window)
        
        # Update viewport to match window size
        viewport = mj.MjrRect(0, 0, width, height)
        
        # Update scene and render
        mj.mjv_updateScene(model, data, self.opt, None, self.cam, 
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)
        
        # Set OpenGL viewport (this is crucial!)
        glfw.make_context_current(self.window)
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_WINDOW.value, self.context)
        
        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def should_close(self):
        return glfw.window_should_close(self.window)

    def close(self):
        glfw.terminate()