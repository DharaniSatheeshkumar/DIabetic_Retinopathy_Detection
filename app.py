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
    ("No DR", "#2E7D5B",
     "No signs of diabetic retinopathy were found in this photo.",
     "Keep up your regular yearly eye check-ups."),
    ("Mild", "#B8860B",
     "A few tiny swellings (microaneurysms) in the retina's blood vessels may be present.",
     "Usually no treatment is needed. Ask an eye doctor for a follow-up in 6-12 months."),
    ("Moderate", "#B85A0B",
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


# ---------------- Result card (matches the redesign) ----------------
INK, SOFT, MUTE, LINE = "#1F2430", "#4A5160", "#5A6270", "#E2D9CC"
CARD = f"background:#fff; border:1px solid {LINE}; border-radius:20px; padding:28px; font-family:Inter,system-ui,sans-serif; color:{INK};"


def placeholder_html():
    return (f'<div style="{CARD} text-align:center; color:{MUTE}; padding:44px 28px;">'
            'Your result will appear here after you upload a photo and click <b>Analyze photo</b>.</div>')


def card_html(p, top):
    name, color, meaning, advice = GRADES[top]
    refer = sum(p[2:])
    if refer >= 0.5:
        flag_bg, flag_txt = "#FBE9E7", "#8A1C1C"
        flag = f"Eye-doctor visit recommended. Chance of Moderate or worse: {refer*100:.0f}%"
    else:
        flag_bg, flag_txt = "#E8F6EE", "#1E6B3F"
        flag = f"Low chance of needing urgent referral. Chance of Moderate or worse: {refer*100:.0f}%"

    bars = ""
    for i, (g, c, _, _) in enumerate(GRADES):
        pct = p[i] * 100
        w = "700" if i == top else "400"
        bars += (
            f'<div style="display:flex; align-items:center; gap:12px; margin:10px 0; font-size:14px; font-weight:{w};">'
            f'<span style="width:104px; flex:none;">{g}</span>'
            '<div style="flex:1; height:12px; border-radius:6px; background:#EEE7DB; overflow:hidden;">'
            f'<div style="width:{max(pct, 0.8):.1f}%; height:100%; background:{c};"></div></div>'
            f'<span style="width:44px; text-align:right; flex:none;">{pct:.0f}%</span></div>'
        )

    note = ""
    if p[top] < 0.5:
        note = (f'<div style="margin-top:14px; font-size:13px; color:{MUTE};">The model is not very sure about this photo '
                '(top confidence below 50%). Treat the result with extra caution.</div>')

    return (
        f'<div style="{CARD}">'
        '<div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">'
        f'<span style="background:{color}; color:#fff; font-weight:700; font-size:22px; padding:8px 22px; border-radius:999px;">{name}</span>'
        f'<span style="font-size:15px; color:{SOFT};">Confidence <b style="color:{INK};">{p[top]*100:.0f}%</b></span></div>'
        f'<p style="margin:20px 0 0; font-size:16px; line-height:1.6;"><b>What it means:</b> {meaning}</p>'
        f'<p style="margin:10px 0 0; font-size:16px; line-height:1.6;"><b>Suggested next step:</b> {advice}</p>'
        f'<div style="margin-top:18px; padding:12px 16px; border-radius:12px; background:{flag_bg}; color:{flag_txt}; font-size:14px; font-weight:600;">{flag}</div>'
        f'<div style="margin-top:26px; font-size:12px; font-weight:700; letter-spacing:.08em; color:{MUTE};">CONFIDENCE FOR EACH GRADE</div>'
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
F = "font-family:Inter,system-ui,sans-serif;"
HEADER = (f'<div style="display:flex; align-items:center; gap:10px; padding:8px 0; {F} font-weight:700; font-size:18px; color:{INK};">'
          '<svg width="30" height="30" viewBox="0 0 30 30" fill="none"><circle cx="15" cy="15" r="13" stroke="#385EA7" stroke-width="2"/>'
          '<circle cx="15" cy="15" r="5.5" fill="#385EA7"/></svg>RetinaScreen</div>')

HERO = (f'<div style="{F} padding:24px 8px 8px 0; color:{INK};">'
        '<div style="font-size:12px; font-weight:700; letter-spacing:.08em; color:#385EA7;">AI-ASSISTED SCREENING</div>'
        '<h1 style="margin:16px 0 0; font-size:46px; line-height:1.1; font-weight:650; letter-spacing:-.025em;">Check a retina photo in seconds.</h1>'
        f'<p style="margin:20px 0 0; font-size:17px; line-height:1.6; color:{SOFT}; max-width:480px;">Upload a fundus photo and get an estimate of the '
        'diabetic retinopathy stage, explained in plain language, with a heatmap of where the model looked.</p>'
        '<div style="display:flex; gap:26px; margin-top:28px; flex-wrap:wrap; font-size:15px; font-weight:600;">'
        '<span><span style="color:#385EA7;">01</span>&nbsp; Upload</span><span><span style="color:#385EA7;">02</span>&nbsp; Analyze</span>'
        '<span><span style="color:#385EA7;">03</span>&nbsp; Read the result</span></div></div>')

HINT = f'<div style="{F} font-size:13px; color:{MUTE};">No photo handy? Pick a sample under the result section.</div>'


def section(title, sub=""):
    return (f'<div style="{F} display:flex; align-items:baseline; justify-content:space-between; flex-wrap:wrap; gap:8px; margin:56px 0 8px;">'
            f'<h2 style="margin:0; font-size:30px; font-weight:650; letter-spacing:-.02em; color:{INK};">{title}</h2>'
            f'<span style="font-size:14px; color:{MUTE};">{sub}</span></div>')


LEGEND = (f'<div style="{F} padding:4px 4px 0;">'
          '<div style="height:8px; border-radius:4px; background:linear-gradient(90deg,#2B50AA,#35A7C9,#FFE45C,#FF7A2B,#D81F3A);"></div>'
          f'<div style="display:flex; justify-content:space-between; font-size:12px; color:{MUTE}; margin-top:6px;"><span>Less influence</span><span>More influence</span></div>'
          f'<p style="margin:12px 0 0; font-size:14px; line-height:1.55; color:{SOFT};">Warm areas influenced the prediction most. '
          'This shows where the model looked, not a confirmed lesion.</p></div>')

rows = "".join(
    f'<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px 32px; padding:18px 0; '
    f'border-top:1px solid #DDD3C4; font-size:15px; line-height:1.5; color:#3A4150;">'
    f'<b style="color:{g[1]};">{g[0]}</b><span>{g[2]}</span><span>{g[3]}</span></div>' for g in GRADES)
GRADES_HTML = section("Understanding the grades") + f'<div style="{F} border-bottom:1px solid #DDD3C4;">{rows}</div>'

FOOTER = (f'<div style="{F} padding:40px 0 24px;">'
          f'<p style="margin:0; font-size:15px; line-height:1.6; color:#3A4150; max-width:760px;"><b>Screening aid only. This is not a medical diagnosis.</b> '
          'Only a qualified eye doctor can diagnose diabetic retinopathy. Do not upload real patient photos to a public site.</p>'
          f'<p style="margin:12px 0 0; font-size:13px; color:{MUTE};">EfficientNet-B0 trained on about 35,000 EyePACS photos at {IMG_SIZE} x {IMG_SIZE} pixels '
          f'&middot; validation kappa {KAPPA_TEXT} &middot; Grad-CAM from the last convolution layer</p></div>')

CSS = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important;}
.gradio-container button.primary {border-radius: 999px !important;}
"""
LIGHT = {"body_background_fill": "#F6F1EA", "background_fill_primary": "#FFFFFF", "background_fill_secondary": "#F6F1EA",
         "block_background_fill": "#FFFFFF", "block_border_color": "#E2D9CC", "border_color_primary": "#E2D9CC",
         "body_text_color": "#1F2430", "block_label_text_color": "#4A5160", "button_primary_background_fill": "#385EA7",
         "button_primary_background_fill_hover": "#2F4F8E", "button_primary_text_color": "#FFFFFF",
         "button_secondary_background_fill": "#FFFFFF", "button_secondary_text_color": "#1F2430", "block_shadow": "none"}
THEME = gr.themes.Base(
    primary_hue="blue", neutral_hue="stone", radius_size="lg",
).set(**LIGHT, **{k + "_dark": v for k, v in LIGHT.items()})  # same light look in dark mode

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
with gr.Blocks(title="RetinaScreen - AI Diabetic Retinopathy Detection", **blocks_kwargs) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            gr.HTML(HERO)
        with gr.Column(scale=5):
            inp = gr.Image(type="pil", label="Retina photo", height=300)
            with gr.Row():
                clear_btn = gr.Button("Clear")
                run_btn = gr.Button("Analyze photo", variant="primary", scale=2)
            gr.HTML(HINT)

    gr.HTML(section("Result", "Updates after you click Analyze photo"))
    with gr.Row(equal_height=False):
        with gr.Column(scale=6):
            out_card = gr.HTML(placeholder_html())
        with gr.Column(scale=4):
            out_cam = gr.Image(label="Where the model looked (Grad-CAM)", height=320, interactive=False)
            gr.HTML(LEGEND)

    if example_paths:
        gr.Examples(
            examples=[[p] for p in example_paths], inputs=inp,
            outputs=[out_card, out_cam], fn=predict,
            run_on_click=True, cache_examples=False,
            label="Try a sample (one click)",
        )
    else:
        gr.HTML(f'<div style="{F} font-size:13px; color:{MUTE};">Tip: put a few retina photos in an <code>examples</code> folder next to this file for one-click samples.</div>')

    gr.HTML(GRADES_HTML)
    gr.HTML(FOOTER)

    run_btn.click(predict, inp, [out_card, out_cam])
    clear_btn.click(clear_all, None, [inp, out_card, out_cam])

if __name__ == "__main__":
    demo.launch(**launch_kwargs)
