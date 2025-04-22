import numpy as np
import cv2 as cv
from PIL import Image

class OpenCvCanvas:
    COLOUR_KEY_MAPS = {"b":(0,0,0), "w":(255,255,255), "r":(0,0,255), "g":(0,255,0), "n":(19,69,139)}
    RADIUS_KEY_MAPS = {"1":1, "2":5, "3":10, "4":20, "5":40}
    MASK_VALUE = (255,255,255)

    def __init__(self, image_filename, size=(400,400), mask_padding=0):
        self.base = cv.resize(cv.imread(image_filename), size)
        self.painted_base = self.base.copy()
        self.mask = np.zeros((*size, 3), np.uint8)
        self.size = size
        self.brush_radius = 20
        self.brush_value = (255,255,255)
        self.is_lmouse_down = False
        self.mask_padding = mask_padding

    def create_window(self, window_name="CV2Window", return_pil = True):
        self.window_name = window_name
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.handle_mouse)

        while True:
            cv.imshow(window_name, self.painted_base)
            k = cv.waitKey(1) & 0xFF # bitmask
            if k == 27:
                break
            else:
                self.handle_key(k)
        cv.destroyAllWindows()

        if return_pil:
            return self.get_base_painted_pil(), self.get_mask_pil()
        return self.painted_base, self.mask


    def handle_mouse(self, event, x, y, flags, param):

            if event == cv.EVENT_LBUTTONDOWN:
                self.is_lmouse_down=True
                self.draw(x,y)
                
            elif event == cv.EVENT_LBUTTONUP:
                self.is_lmouse_down = False

            elif event == cv.EVENT_MOUSEMOVE and self.is_lmouse_down:
                self.draw(x,y)


    def draw(self, x, y, value=(255,255,255)):
        # draw onto painted base 
        cv.circle(self.painted_base, (x,y), self.brush_radius, self.brush_value, -1)
        # draw onto the mask the same amount in white
        cv.circle(self.mask, (x,y), self.brush_radius+self.mask_padding, OpenCvCanvas.MASK_VALUE, -1)

    def handle_key(self, key):
        if key in [ord(k) for k in OpenCvCanvas.COLOUR_KEY_MAPS.keys()]:
            self.brush_value = OpenCvCanvas.COLOUR_KEY_MAPS[chr(key)]

        elif key in [ord(k) for k in OpenCvCanvas.RADIUS_KEY_MAPS.keys()]:
            self.brush_radius = OpenCvCanvas.RADIUS_KEY_MAPS[chr(key)]

    def get_base(self):
        return self.base
    
    def get_base_painted(self):
        return self.painted_base
    
    def get_mask(self):
        return self.mask
    
    def get_base_pil(self):
        return Image.fromarray(self.base[..., ::-1]).convert("RGB")
    
    def get_base_painted_pil(self):
        return Image.fromarray(self.painted_base[..., ::-1]).convert("RGB")
    
    def get_mask_pil(self):
        return Image.fromarray(self.mask[..., ::-1]).convert("L")


import torch
from diffusers import AutoPipelineForInpainting

def generate_image(mask_pil, img_pil, size = (400,400), prompt="head of dragon", params = {}):

    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", 
        torch_dtype=torch.float16, 
        variant="fp16",
        safety_checker=None
        
    ).to("cuda")
    # pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    # pipeline.enable_xformers_memory_efficient_attention()

    image = pipeline(prompt=prompt, 
                     image=img_pil, 
                     mask_image=mask_pil, 
                     width=size[0], 
                     height=size[1], 
                    # #  blur_factor=20,
                    **params
                     ).images[0]
    # make_image_grid([imgi, mask_pil, image], rows=1, cols=3)
    return image 