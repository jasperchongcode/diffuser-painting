# diffuser-painting

An interactive Jupyter Notebook interface for hand-drawn inpainting with Stable Diffusion. Draw a basic sketch (which auto generates a mask), then generate AI-enhanced images based on your sketch + prompt.

---

## 🚀 Features

- **Draw Mode**: Freehand sketch with adjustable brush size, color, opacity, and eraser  
- **Mask Padding**: Automatically expands your mask by a configurable number of pixels for better diffusion context  
- **Generate Mode**: Inpaint masked regions guided by your text prompt  
- **Hyperparameter Tuning**: Control strength, guidance scale, inference steps, negative prompts, and upscaling factor for fine-tuned results  

---

## 📦 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/diffusion-drawing-playground.git
   cd diffusion-drawing-playground
   ```
2. **(Optional) Create & activate Conda env**  
   ```bash
   conda create -n draw-diffuse python=3.9
   conda activate draw-diffuse
   ```
3. **Install dependencies**  (haven't checked that this works)
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch Notebook**  
   ```bash
   jupyter notebook
   ```

---

## 🎨 Draw

1. Open **main.ipynb**  
2. Run the **Draw** cell to open the canvas  
3. Use the top-bar keybinds:
   - **Color Picker**: Change brush color  
   - **Size Slider**: Adjust brush diameter  
   - **Opacity Slider**: Control brush opacity  
   - **Mode Toggle**: Switch between **Draw** and **Erase**

> **Mask Padding**  
> The `mask_padding` parameter expands your drawn mask by *n* pixels. This gives the diffusion model extra “context” around masked regions, improving inpaint quality.

> **Eraser Note**  
> Erasing removes both mask and padding. To preserve padded edges after erasing, re-draw over the erased border.

---

## ✨ Generate

1. Enter your **text prompt** in the next cell  
2. Adjust hyperparameters to taste  
3. Run the cell; rerun for variations

### Hyperparameter Tuning

| Parameter             | Description                                                                                                                                                           | Typical Range |
|-----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| `strength`            | Noise added to the mask.<br>**0.0** → fully rely on your sketch<br>**1.0** → fully rely on the model’s imagination                                                     | 0.4–0.9       |
| `guidance_scale`      | How closely the model follows your text prompt. Too high → “frying” artifacts                                                                                          | 7.5–12.5      |
| `num_inference_steps` | More steps → finer detail (with diminishing returns)                                                                                                                   | 25–75         |
| `negative_prompt`     | Terms to avoid (e.g., “blurry, deformed”)                                                                                                                              | —             |
| `upscaling_factor`    | Render at higher resolution before downsampling for sharper results                                                                                                  | 1.0–2.0       |


## 📝 Example Workflow

1. Set `mask_padding = 20` for extra context (can also try 10 etc.) 
2. **Draw** your best dragon head over an image of dogs
3. Prompt:  
   ```
   A fierce red dragon head with curved horns, emerging from swirling mist.
   ```
4. Tune: `strength = 0.5`, `guidance_scale = 12.5`, `num_inference_steps = 50`  
5. **Generate** → admire your AI-enhanced sketch!

---
