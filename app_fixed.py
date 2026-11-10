import os, glob, inspect, traceback
import numpy as np
import torch, timm, gradio as gr
from PIL import Image
from torchvision import transforms as T

# ---------------- Settings ----------------
IMG_SIZE = 384
MODEL_FILE = "dr_best_384.pth"
MODEL_NAME = "efficientnet_b0"
KAPPA_TEXT = "0.72"
HERE = os.path.dirname(os.path.abspath(__file__))

# name, color, meaning, next step
GRADES = [
    ("No DR", "#2E9E5B",
     "No signs of diabetic retinopathy were found in this photo.",
     "Keep up your regular yearly eye check-ups."),
    ("Mild", "#D9A400",
     "A few tiny swellings (microaneurysms) in the retina's blood vessels may be present.",
     "Usually no treatment is needed. Ask an eye doctor for a follow-up in 6-12 months."),
    ("Moderate", "#EE8A1B",
     "More blood vessels are affected, with possible small bleeds or fluid leaks.",
     "See an eye doctor soon. Treatment and closer monitoring may be needed."),
    ("Severe", "#DC3545",
     "Many blood vessels are blocked, so parts of the retina are short of blood.",
     "See an eye specialist promptly. The risk of the disease getting worse is high."),
    ("Proliferative", "#A71D2A",
     "The most advanced stage: fragile new blood vessels are growing and can bleed.",
     "Urgent: see an eye specialist as soon as possible. Treatment can protect vision."),
]

# ---------------- Model ----------------
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=5)
model.load_state_dict(torch.load(os.path.join(HERE, MODEL_FILE), map_location="cpu"))
model.eval()

tf = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ---------------- Image helpers ----------------
def crop_black(img, thr=12):
    """Trim the black border around the retina (training images were cropped the same way)."""
    a = np.asarray(img.convert("L"))
    ys, xs = np.where(a > thr)
    if len(ys) < 100:
        return img
    return img.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))


def pad_square(img):
    w, h = img.size
    m = max(w, h)
    canvas = Image.new("RGB", (m, m), (0, 0, 0))
    canvas.paste(img, ((m - w) // 2, (m - h) // 2))
    return canvas


def prepare(img):
    img = crop_black(img.convert("RGB"))
    return pad_square(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def colorize(cam):
    """Simple jet-style colormap (no matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def gradcam_overlay(disp, x, class_idx):
    """Grad-CAM on the last conv feature map of EfficientNet."""
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        feats = model.forward_features(x)
        feats.retain_grad()
        logits = model.forward_head(feats)
        logits[0, class_idx].backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * feats).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    base = np.asarray(disp).astype(np.float32) / 255.0
    alpha = (cam[..., None] ** 0.8) * 0.6
    out = base * (1 - alpha) + colorize(cam) * alpha
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


# ---------------- Result card ----------------
TXT = "color: var(--body-text-color);"
MUTED = "color: var(--body-text-color-subdued);"
PANEL = "background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); border-radius: 14px;"


def placeholder_html():
    return (f'<div style="{PANEL} padding: 28px; text-align: center; {MUTED}">'
            'Your result will appear here after you upload a photo and click <b>Analyze photo</b>.</div>')


def card_html(p, top):
    name, color, meaning, advice = GRADES[top]
    refer = sum(p[2:])
    if refer >= 0.5:
        flag_bg, flag_txt = "#FDECEA", "#8A1C1C"
        flag = f"Eye-doctor visit recommended. Chance of Moderate or worse: {refer*100:.0f}%"
    else:
        flag_bg, flag_txt = "#E8F6EE", "#1E6B3F"
        flag = f"Low chance of needing urgent referral. Chance of Moderate or worse: {refer*100:.0f}%"

    bars = ""
    for i, (g, c, _, _) in enumerate(GRADES):
        pct = p[i] * 100
        weight = "700" if i == top else "400"
        bars += (
            '<div style="display:flex; align-items:center; gap:10px; margin:7px 0;">'
            f'<div style="width:105px; font-size:14px; font-weight:{weight}; {TXT}">{g}</div>'
            '<div style="flex:1; height:14px; border-radius:7px; background: var(--border-color-primary); overflow:hidden;">'
            f'<div style="width:{max(pct, 0.8):.1f}%; height:100%; background:{c}; border-radius:7px;"></div></div>'
            f'<div style="width:48px; text-align:right; font-size:14px; font-weight:{weight}; {TXT}">{pct:.0f}%</div>'
            '</div>'
        )

    note = ""
    if p[top] < 0.5:
        note = (f'<div style="margin-top:12px; font-size:13px; {MUTED}">'
                'The model is not very sure about this photo (top confidence below 50%). '
                'Treat the result with extra caution.</div>')

    return (
        f'<div style="{PANEL} padding: 22px;">'
        '<div style="display:flex; align-items:center; gap:14px; flex-wrap:wrap;">'
        f'<div style="background:{color}; color:#fff; font-weight:700; font-size:20px; padding:8px 18px; border-radius:999px;">{name}</div>'
        f'<div style="font-size:15px; {MUTED}">Confidence <b style="{TXT}">{p[top]*100:.0f}%</b></div>'
        '</div>'
        f'<div style="margin-top:16px; font-size:15px; line-height:1.55; {TXT}"><b>What it means:</b> {meaning}</div>'
        f'<div style="margin-top:8px; font-size:15px; line-height:1.55; {TXT}"><b>Suggested next step:</b> {advice}</div>'
        f'<div style="margin-top:14px; padding:10px 14px; border-radius:10px; background:{flag_bg}; color:{flag_txt}; font-size:14px; font-weight:600;">{flag}</div>'
        f'<div style="margin-top:18px; font-size:13px; font-weight:700; letter-spacing:.04em; {MUTED}">CONFIDENCE FOR EACH GRADE</div>'
        f'{bars}{note}</div>'
    )


# ---------------- Prediction ----------------
def predict(img):
    if img is None:
        return placeholder_html(), None
    disp = prepare(img)
    x = tf(disp).unsqueeze(0)
    with torch.no_grad():
        p = torch.softmax(model(x), 1)[0].tolist()
    top = max(range(5), key=lambda i: p[i])
    try:
        cam_img = gradcam_overlay(disp, x, top)
    except Exception:
        traceback.print_exc()
        cam_img = None
    return card_html(p, top), cam_img


def clear_all():
    return None, placeholder_html(), None


# ---------------- Static text ----------------
BANNER = (
    '<div style="background: linear-gradient(120deg, #12305a 0%, #2A6BAF 100%); border-radius: 16px; '
    'padding: 28px 26px; text-align: center;">'
    '<div style="color:#fff; font-size: 30px; font-weight: 800; letter-spacing: .01em;">AI Diabetic Retinopathy Detection</div>'
    '<div style="color:#dbe9fb; font-size: 15px; margin-top: 8px;">Upload a retina photo to get an instant estimate of the '
    'diabetic retinopathy stage, explained in plain language.</div></div>'
)

HOW_TO = """
### How to use
1. **Upload** a clear retina (fundus) photo, or click one of the **samples**.
2. Click **Analyze photo**.
3. Read the **result card** and check the **heatmap** to see where the model looked.

**Best photos:** sharp, well lit, centered on the retina, no blur or heavy glare.
"""

CAM_NOTE = ("*Warm colors (red and yellow) are the areas that influenced the prediction most. "
            "This shows where the model looked, not a confirmed lesion.*")

GRADE_TABLE = """
### Understanding the grades
| Grade | What it means | Typical next step |
|---|---|---|
""" + "\n".join(f"| **{g[0]}** | {g[2]} | {g[3]} |" for g in GRADES)

ABOUT = f"""
- **Model:** EfficientNet-B0 trained on about 35,000 retina photos from the EyePACS dataset at {IMG_SIZE}x{IMG_SIZE} pixels.
- **Validation score:** quadratic weighted kappa of {KAPPA_TEXT}. Mild cases are the hardest to detect.
- **Confidence bars:** show how strongly the model leans toward each grade. They are not a guarantee.
- **Heatmap:** Grad-CAM, computed from the model's last convolution layer.
"""

DISCLAIMER = (
    "**Screening aid only. This is not a medical diagnosis.** "
    "Only a qualified eye doctor can diagnose diabetic retinopathy. "
    "Do not upload real patient photos to a public site."
)

CSS = """
.gradio-container {max-width: 1080px !important; margin: 0 auto !important;}
"""
THEME = gr.themes.Soft(
    primary_hue="blue", secondary_hue="sky", neutral_hue="slate",
)

# Gradio 6 takes theme/css in launch(); older versions take them in Blocks().
if "theme" in inspect.signature(gr.Blocks.launch).parameters:
    blocks_kwargs, launch_kwargs = {}, {"theme": THEME, "css": CSS}
else:
    blocks_kwargs, launch_kwargs = {"theme": THEME, "css": CSS}, {}

example_paths = sorted(
    p for ext in ("*.jpg", "*.jpeg", "*.png")
    for p in glob.glob(os.path.join(HERE, "examples", ext))
)

# ---------------- Layout ----------------
with gr.Blocks(title="AI Diabetic Retinopathy Detection", **blocks_kwargs) as demo:
    gr.HTML(BANNER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            inp = gr.Image(type="pil", label="Retina photo", height=340)
            with gr.Row():
                clear_btn = gr.Button("Clear")
                run_btn = gr.Button("Analyze photo", variant="primary", scale=2)
        with gr.Column(scale=4):
            gr.Markdown(HOW_TO)

    gr.Markdown("## Result")
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            out_card = gr.HTML(placeholder_html())
        with gr.Column(scale=4):
            out_cam = gr.Image(label="Where the model looked (Grad-CAM)", height=340,
                               interactive=False)
            gr.Markdown(CAM_NOTE)

    if example_paths:
        gr.Examples(
            examples=[[p] for p in example_paths], inputs=inp,
            outputs=[out_card, out_cam], fn=predict,
            run_on_click=True, cache_examples=False,
            label="Try a sample (one click)",
        )
    else:
        gr.Markdown("*Tip: put a few retina photos in an `examples` folder next to this file to get one-click samples.*")

    gr.Markdown(GRADE_TABLE)
    with gr.Accordion("About this model", open=False):
        gr.Markdown(ABOUT)
    gr.Markdown(DISCLAIMER)

    run_btn.click(predict, inp, [out_card, out_cam])
    clear_btn.click(clear_all, None, [inp, out_card, out_cam])

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
