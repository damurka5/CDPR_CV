from mujoco.glfw import glfw
import mujoco as mj

class CDPRViewer:
    def __init__(self, model, width=1200, height=900):
        self.window = self._init_glfw(width, height)
        self.scene, self.context, self.cam, self.opt = self._init_viewer(model)
        
    def _init_glfw(self, width, height):
        glfw.init()
        window = glfw.create_window(width, height, "CDPR Simulation", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)
        return window
        
    def _init_viewer(self, model):
        cam = mj.MjvCamera()
        opt = mj.MjvOption()
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        scene = mj.MjvScene(model, maxgeom=10000)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
        return scene, context, cam, opt
        
    def update(self, model, data):
        viewport = self._get_viewport()
        mj.mjv_updateScene(model, data, self.opt, None, self.cam, 
                          mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def _get_viewport(self):
        width, height = glfw.get_framebuffer_size(self.window)
        return mj.MjrRect(0, 0, width, height)
        
    def should_close(self):
        return glfw.window_should_close(self.window)
        
    def close(self):
        glfw.terminate()