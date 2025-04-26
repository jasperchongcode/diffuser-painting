from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers import AutoPipelineForInpainting
import torch
import numpy as np
import cv2 as cv
from PIL import Image, ImageDraw, ImageFont


class CVPaint:
    # constants defining keybinds
    COLOUR_KEY_MAPS = {"b": (0, 0, 0), "w": (255, 255, 255), "r": (0, 0, 255), "g": (
        0, 255, 0), "l": (255, 0, 0), "n": (19, 69, 139), "o": (0, 165, 255),
        "p":(128,0,128)}
    RADIUS_KEY_MAPS = {"1": 1, "2": 5, "3": 10, "4": 20, "5": 40}
    ALPHA_KEY_MAPS = {"6": 1, "7": 0.8, "8": 0.6, "9": 0.4, "0": 0.2}
    ERASER_MODE_KEY = "e"
    # constant defining the mask value when drawing in 
    MASK_VALUE = (255, 255, 255)

    def __init__(self, image_filename, size=(400, 400), mask_padding=10):
        """Initialise CVPaint with image_filename (optional: size and mask padding)"""
        self.base = cv.resize(cv.imread(image_filename), size) # base image - never changed
        self.painted_base = self.base.copy() # Version of the base image to append to 
        self.display_base = self.painted_base.copy() # image to display (draw temp buffer and cursor on it)
        self.mask = np.zeros((*size, 3), np.uint8) # initialise mask as full black
        # this is so you can correctly do strokes without stacking transparency
        self.stroke_buffer = np.ones((*size, 3), dtype=np.float32)*-1
        self.size = size # image size
        self.brush_radius = 20 # radius of circular brush
        self.brush_value = (255, 255, 255) # start with white brush
        self.alpha = 1 # 1 - no transparency, 0 - full transparency
        self.is_lmouse_down = False
        self.mask_padding = mask_padding # how many pixels wider than the drawing should the mask be (gives space for the model to generate)
        self.eraser_mode = False # by default in drawing mode

        self.info_bar_height = 40 # height of bar at top to display keybinds
        self.create_info_bar() # build it

    def create_window(self, window_name="CVPaint", return_pil=True):
        """Main control loop"""
        self.window_name = window_name # assuming this is never changed, just a reference
        cv.namedWindow(window_name, cv.WINDOW_GUI_NORMAL)
        cv.resizeWindow(window_name, (self.size[0], self.size[1]+self.info_bar_height))

        cv.setMouseCallback(window_name, self.handle_mouse,)

        while True:
            # draw buffer temporarily
            cv.imshow(window_name, np.concatenate(
                [self.info_bar, self.display_base], axis=0))
            k = cv.waitKey(1) & 0xFF  # bitmask
            if k == 27:
                break
            else:
                self.handle_key(k)
        cv.destroyAllWindows()

        if return_pil:
            return self.get_base_painted_pil(), self.get_mask_pil()
        return self.painted_base, self.mask

    def handle_mouse(self, event, x, y, flags, param):
        """Logic for handling mouse events"""
        y -= self.info_bar_height

        if y < 0:
            # print("Attempting to draw on info bar")
            return

        self.draw_ghost_cursor(x, y)
        if (np.any(self.stroke_buffer != -1)):  # display the temporary buffer if it exists
            self.draw_stroke_buffer()
            
        if event == cv.EVENT_LBUTTONDOWN:
            self.is_lmouse_down = True
            self.draw(x, y)

        elif event == cv.EVENT_LBUTTONUP:
            self.is_lmouse_down = False
            if (np.any(self.stroke_buffer != -1)):
                self.clear_buffer()

        elif event == cv.EVENT_MOUSEMOVE and self.is_lmouse_down:
            self.draw(x, y)

    def clear_buffer(self):
        """Draws the stroke buffer onto painted_base and clears it (based off of alpha at the time of calling)."""
        # convert types so it doesnt freak out with -1
        stroke_float = self.stroke_buffer.astype(np.float32)
        painted_float = self.painted_base.astype(np.float32)
        # combine them
        blended = cv.addWeighted(
            stroke_float, self.alpha, painted_float, 1 - self.alpha, 0)
        # Conditionally add the blended ones when its not -1
        self.painted_base = np.where(
            self.stroke_buffer[:, :, :] == -1, self.painted_base, blended.astype(np.uint8))
        # clear buffer
        self.stroke_buffer = np.ones((*self.size, 3), dtype=np.float32)*-1
        # print("CLEARED STROKE BUFFER:", (np.any(self.stroke_buffer != -1)))

    def draw(self, x, y, value=(255, 255, 255)):
        """Logic for drawing onto the painted base/saving to the stroke buffer (if alpha < 1).
        Also draws onto the mask."""
        # eraser 
        if self.eraser_mode:
            erase_mask = np.zeros(self.size, dtype=np.uint8)
            cv.circle(erase_mask, (x, y), self.brush_radius, 1, -1)
            # apply the original image based on the mask
            self.painted_base[erase_mask==1] = self.base[erase_mask==1]

            # subtract from the mask the same amount + padding
            cv.circle(self.mask, (x, y), self.brush_radius +
                    self.mask_padding, (0, 0, 0), -1)

        else: # draw more
            if self.alpha < 1:
                # save to buffer so dont stack transparency
                cv.circle(self.stroke_buffer, (x, y),
                        self.brush_radius, self.brush_value, -1)
            else:
                # draw onto painted base
                cv.circle(self.painted_base, (x, y),
                        self.brush_radius, self.brush_value, -1)
            # draw onto the mask the same amount + mask_padding in white
            cv.circle(self.mask, (x, y), self.brush_radius +
                    self.mask_padding, CVPaint.MASK_VALUE, -1)

    def handle_key(self, key):
        """Logic for handling keypresses based off of constants at start of class (not including ESC)"""
        key = key
        
        if key in [ord(k) for k in CVPaint.COLOUR_KEY_MAPS.keys()]:
            self.brush_value = CVPaint.COLOUR_KEY_MAPS[chr(key)]

        elif key in [ord(k) for k in CVPaint.RADIUS_KEY_MAPS.keys()]:
            self.brush_radius = CVPaint.RADIUS_KEY_MAPS[chr(key)]

        elif key in [ord(k) for k in CVPaint.ALPHA_KEY_MAPS.keys()]:
            self.alpha = CVPaint.ALPHA_KEY_MAPS[chr(key)]

        elif key == ord(CVPaint.ERASER_MODE_KEY):
            self.eraser_mode = not self.eraser_mode

    def create_info_bar(self, background=(255, 255, 255)):
        """Set self.info_bar to image_width*info_bar_height image displaying keybinds"""
        info_bar = np.full(
            (self.info_bar_height, self.size[0], 3), background, dtype=np.uint8)
        
        outline_colour = (0, 0, 0) # some colour thats visible on the background colour
        letter_size = 2
        letter_spacing = 20 
        initial_spacing = 0
        line_spacing = 1

        # display colour keybinds
        for i, key in enumerate(CVPaint.COLOUR_KEY_MAPS):
            cv.putText(info_bar, str(key).upper(), (i*letter_spacing + initial_spacing, int(
                self.info_bar_height/1.5)), cv.FONT_HERSHEY_SIMPLEX, 1, outline_colour, letter_size+1, cv.LINE_AA)
            cv.putText(info_bar, str(key).upper(), (i*letter_spacing + initial_spacing, int(self.info_bar_height/1.5)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, CVPaint.COLOUR_KEY_MAPS[key], letter_size, cv.LINE_AA)
        ending_spacing = (i+1)*letter_spacing
        
        # vertical line
        ending_spacing += line_spacing
        cv.line(info_bar, (ending_spacing, 0), (ending_spacing,
                self.info_bar_height), outline_colour, 2)
        ending_spacing += line_spacing

        # display size keybinds
        for i, key in enumerate(CVPaint.RADIUS_KEY_MAPS):
            cv.putText(info_bar, str(key), (i*letter_spacing + ending_spacing, int(self.info_bar_height/1.5)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, outline_colour, int(CVPaint.RADIUS_KEY_MAPS[key]/4), cv.LINE_AA)
            # cv.circle(info_bar, (i*letter_spacing + ending_spacing, int(height/1.5)), int(OpenCvCanvas.RADIUS_KEY_MAPS[key]), outline_colour, -1)
        ending_spacing += (i+1)*letter_spacing
        
        # vertical line
        ending_spacing += line_spacing
        cv.line(info_bar, (ending_spacing, 0), (ending_spacing,
                self.info_bar_height), outline_colour, 2)
        ending_spacing += line_spacing

        # display transparency keybinds
        for i, key in enumerate(CVPaint.ALPHA_KEY_MAPS):
            key_colour = (0+255*(1-CVPaint.ALPHA_KEY_MAPS[key]))
            cv.putText(info_bar, str(key), (i*letter_spacing + ending_spacing, int(self.info_bar_height/1.5)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (key_colour, key_colour, key_colour), letter_size, cv.LINE_AA)
        ending_spacing += (i+1)*letter_spacing
                
        # vertical line
        ending_spacing += line_spacing
        cv.line(info_bar, (ending_spacing, 0), (ending_spacing,
                self.info_bar_height), outline_colour, 2)
        ending_spacing += line_spacing
        
        # display eraser
        cv.putText(info_bar, CVPaint.ERASER_MODE_KEY.upper(), (ending_spacing, int(
                self.info_bar_height/1.5)), cv.FONT_HERSHEY_SIMPLEX, 1, outline_colour, 3, cv.LINE_AA)
        cv.putText(info_bar, CVPaint.ERASER_MODE_KEY.upper(), (ending_spacing, int(self.info_bar_height/1.5)),
                cv.FONT_HERSHEY_SIMPLEX, 1, (157, 161, 245), 2, cv.LINE_AA)
            

        self.info_bar = info_bar

    def draw_ghost_cursor(self, x, y):
        """Draw the cursor indicating brush size, colour, opacity. Also draws the temporary buffer."""
        # print("drawing ghost ")
        self.display_base = self.painted_base.copy()
        if self.eraser_mode: # show eraser cursor
           # mask to show the original image as the cursor
            eraser_mask = np.zeros(self.size, np.uint8)
            cv.circle(eraser_mask, (x, y), self.brush_radius, 1, 1)
            # Add a little bit of white to the original pixels so can see cursor on unpainted
            white_overlay = np.ones((*self.size, 3), np.uint8)*255
            cursor_visibility = 0.4
            transparent_base = cv.addWeighted(white_overlay, cursor_visibility, self.base, 1-cursor_visibility, 0) # make a bit more white than original base so stands out
            self.display_base[eraser_mask==1] = transparent_base[eraser_mask==1]

        else: # standard drawing mode
            if self.alpha < 1:
                # draw circle with transparency
                overlay = self.display_base.copy()
                cv.circle(overlay, (x, y), self.brush_radius, self.brush_value, 1)
                self.display_base = cv.addWeighted(
                    overlay, self.alpha, self.display_base, 1-self.alpha, 0)
                del overlay
            else:
                cv.circle(self.display_base, (x, y),
                        self.brush_radius, self.brush_value, 1)
                
    def draw_stroke_buffer(self):
        """Draw the stroke buffer onto self.display_base"""
        stroke_float = self.stroke_buffer.astype(np.float32)
        display_float = self.display_base.astype(np.float32)
        blended = cv.addWeighted(
            stroke_float, self.alpha, display_float, 1 - self.alpha, 0)
        self.display_base = np.where(
            self.stroke_buffer[:, :, :] == -1, self.display_base, blended.astype(np.uint8))

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


def generate_image(mask_pil, img_pil, size=(400, 400), prompt="head of dragon", params={}):

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


def generate_image_controlnet(mask_pil, img_pil, size=(400, 400), prompt="head of dragon", params={}):
    print("Begin generating image")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant="fp16")

    pipeline = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None

    ).to("cuda")

    print("Loaded pipeline")
    # pipeline.safety_checker = lambda images, clip_input: (images, [False] * len(images))
    # pipeline.enable_xformers_memory_efficient_attention()

    def make_inpaint_condition(init_image, mask_image):
        init_image = np.array(init_image.convert(
            "RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert(
            "L")).astype(np.float32) / 255.0

        assert init_image.shape[0:1] == mask_image.shape[0:
                                                         1], "image and image_mask must have the same image size"
        init_image[mask_image > 0.5] = -1.0  # set as masked pixel
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image

    control_image = make_inpaint_condition(img_pil, mask_pil)
    print("Created control image")

    image = pipeline(prompt=prompt,
                     image=img_pil,
                     mask_image=mask_pil,
                     control_image=control_image,
                     width=size[0],
                     height=size[1],
                     # #  blur_factor=20,
                     **params
                     ).images[0]
    # make_image_grid([imgi, mask_pil, image], rows=1, cols=3)
    return image


def make_image_row(images: list[Image.Image], labels: list[str] = [], font: ImageFont.FreeTypeFont = ImageFont.truetype("times.ttf", size=24), size = (400,400) ) -> Image.Image:

    if labels and len(images) != len(labels):
        print("ERRROR")

    
    images = [img.resize(size) for img in images]

    output = Image.new("RGB", (size[0]*len(images), size[1]))
    draw = ImageDraw.Draw(output)

    for i, img in enumerate(images):
        output.paste(img, (i*size[0], 0))
        if labels:
            draw.text((i*size[0], 0), labels[i], font=font, fill = (255,255,255), stroke_width=1, stroke_fill=(0,0,0))
    
    return output