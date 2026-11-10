# AI-Powered Diabetic Retinopathy Detection

A final-year project that grades retinal (fundus) photos into five diabetic retinopathy stages and explains the result in plain language, with a Grad-CAM heatmap showing where the model looked.

**Screening aid only. This is not a medical diagnosis.**

## Grades
No DR, Mild, Moderate, Severe, Proliferative (grades 0 to 4).

## Method
- Data: EyePACS training images (Kaggle Diabetic Retinopathy Detection), using the resized and cropped copy `tanlikesmath/diabetic-retinopathy-resized` (35,108 images)
- Preprocessing: pad to a square, resize to 384 x 384, ImageNet normalization, flip, rotation and brightness augmentation
- Model: EfficientNet-B0 (timm, ImageNet pretrained), 5 outputs
- Training: class-weighted cross-entropy, AdamW (lr 3e-4), cosine schedule, 10 epochs, 80/20 stratified split, Google Colab T4 GPU
- App: Gradio, with a result card, per-grade confidence bars and a Grad-CAM heatmap

## Results (validation split)
| Input size | Best quadratic weighted kappa | Accuracy |
|---|---|---|
| 224 x 224 | 0.592 | 0.633 |
| 384 x 384 | 0.719 | 0.714 |

The best epoch was chosen on this same validation split, so the score is slightly optimistic. External validation (MESSIDOR-2 or APTOS 2019) is planned.

## Run locally
```
pip install -r requirements.txt
python app.py
```
Put the trained weights `dr_best_384.pth` next to `app.py`. To get one-click samples, add a few retina photos to an `examples` folder.

## Limitations
- Trained on one dataset, so results may be lower on photos from other cameras or clinics
- Mild cases are the hardest to detect
- Not validated for clinical use

## Credits
EyePACS data via Kaggle; resized copy by the Kaggle user tanlikesmath; PyTorch, timm and Gradio.
