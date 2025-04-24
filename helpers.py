from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers import AutoPipelineForInpainting
import torch
import numpy as np
import cv2 as cv
from PIL import Image


class OpenCvCanvas:
    COLOUR_KEY_MAPS = {"b": (0, 0, 0), "w": (255, 255, 255), "r": (0, 0, 255), "g": (
        0, 255, 0), "l": (255, 0, 0), "n": (19, 69, 139), "o": (0, 165, 255)}
    RADIUS_KEY_MAPS = {"1": 1, "2": 5, "3": 10, "4": 20, "5": 40}
    ALPHA_KEY_MAPS = {"6": 1, "7": 0.8, "8": 0.6, "9": 0.4, "0": 0.2}
    MASK_VALUE = (255, 255, 255)

    def __init__(self, image_filename, size=(400, 400), mask_padding=0):
        self.base = cv.resize(cv.imread(image_filename), size)
        self.painted_base = self.base.copy()
        self.display_base = self.painted_base.copy()
        self.mask = np.zeros((*size, 3), np.uint8)
        # this is so you can correctly do strokes without stacking transparency
        self.stroke_buffer = np.ones((*size, 3), dtype=np.float32)*-1
        self.size = size
        self.brush_radius = 20
        self.brush_value = (255, 255, 255)
        self.alpha = 1
        self.is_lmouse_down = False
        self.mask_padding = mask_padding

        self.info_bar_height = 40
        self.create_info_bar()

    def create_window(self, window_name="CV2Window", return_pil=True):
        self.window_name = window_name
        cv.namedWindow(window_name)
        cv.setMouseCallback(window_name, self.handle_mouse)

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
        y -= self.info_bar_height

        if y < 0:
            # print("Attempting to draw on info bar")
            return

        self.draw_ghost_cursor(x, y)

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
        if self.alpha < 1:
            # save to buffer so dont stack transparency
            cv.circle(self.stroke_buffer, (x, y),
                      self.brush_radius, self.brush_value, -1)
        else:
            # draw onto painted base
            cv.circle(self.painted_base, (x, y),
                      self.brush_radius, self.brush_value, -1)
        # draw onto the mask the same amount in white
        cv.circle(self.mask, (x, y), self.brush_radius +
                  self.mask_padding, OpenCvCanvas.MASK_VALUE, -1)

    def handle_key(self, key):
        if key in [ord(k) for k in OpenCvCanvas.COLOUR_KEY_MAPS.keys()]:
            self.brush_value = OpenCvCanvas.COLOUR_KEY_MAPS[chr(key)]

        elif key in [ord(k) for k in OpenCvCanvas.RADIUS_KEY_MAPS.keys()]:
            self.brush_radius = OpenCvCanvas.RADIUS_KEY_MAPS[chr(key)]

        elif key in [ord(k) for k in OpenCvCanvas.ALPHA_KEY_MAPS.keys()]:
            self.alpha = OpenCvCanvas.ALPHA_KEY_MAPS[chr(key)]

    def create_info_bar(self, background=(255, 255, 255)):
        info_bar = np.full(
            (self.info_bar_height, self.size[0], 3), background, dtype=np.uint8)
        print(info_bar.shape, info_bar.dtype)
        outline_colour = (0, 0, 0)
        letter_spacing = 30
        initial_spacing = 5
        for i, key in enumerate(OpenCvCanvas.COLOUR_KEY_MAPS):
            cv.putText(info_bar, str(key), (i*letter_spacing + initial_spacing, int(
                self.info_bar_height/1.5)), cv.FONT_HERSHEY_SIMPLEX, 1, outline_colour, 3, cv.LINE_AA)
            cv.putText(info_bar, str(key), (i*letter_spacing + initial_spacing, int(self.info_bar_height/1.5)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, OpenCvCanvas.COLOUR_KEY_MAPS[key], 2, cv.LINE_AA)
            ending_spacing = i*letter_spacing+initial_spacing

        ending_spacing += letter_spacing
        cv.line(info_bar, (ending_spacing, 0), (ending_spacing,
                self.info_bar_height), outline_colour, 2)
        ending_spacing += letter_spacing
        for i, key in enumerate(OpenCvCanvas.RADIUS_KEY_MAPS):
            cv.putText(info_bar, str(key), (i*letter_spacing + ending_spacing, int(self.info_bar_height/1.5)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, outline_colour, int(OpenCvCanvas.RADIUS_KEY_MAPS[key]/4), cv.LINE_AA)
            # cv.circle(info_bar, (i*letter_spacing + ending_spacing, int(height/1.5)), int(OpenCvCanvas.RADIUS_KEY_MAPS[key]), outline_colour, -1)

        self.info_bar = info_bar

    def draw_ghost_cursor(self, x, y):
        # print("drawing ghost ")
        self.display_base = self.painted_base.copy()
        if (np.any(self.stroke_buffer != -1)):  # display the temporary buffer if it exists
            # print("DRAWING IN TEMP BUFFER")
            stroke_float = self.stroke_buffer.astype(np.float32)
            display_float = self.display_base.astype(np.float32)
            blended = cv.addWeighted(
                stroke_float, self.alpha, display_float, 1 - self.alpha, 0)
            self.display_base = np.where(
                self.stroke_buffer[:, :, :] == -1, self.display_base, blended.astype(np.uint8))

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
