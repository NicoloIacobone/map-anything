import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

################################################################
# ANALYZER CLASS
################################################################

class TripletAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.student_dir = os.path.join(base_dir, "student")
        self.teacher_dir = os.path.join(base_dir, "teacher")

        self.out_dir = os.path.join(base_dir, "analysis_output")
        os.makedirs(self.out_dir, exist_ok=True)

    ################################################################
    # Utilities
    ################################################################

    def load_embedding(self, path):
        emb = torch.load(path, map_location="cpu")

        if emb.dim() == 4:
            emb = emb[0]  # assume [B,C,H,W]
        elif emb.dim() != 3:
            raise ValueError(f"Unexpected embedding shape: {emb.shape}")

        return emb.float().contiguous()

    def compute_l1(self, teacher, student):
        l1_map = (teacher - student).abs().mean(dim=0)  # [H,W]
        return l1_map, l1_map.mean().item()

    def compute_distribution(self, feat):
        pix = feat.permute(1, 2, 0).reshape(-1, feat.shape[0])  # [H*W, C]
        return pix.mean().item(), pix.std().item(), pix.flatten().numpy()

    def save_heatmap(self, l1_map, out_path):
        plt.figure(figsize=(6, 6))
        plt.imshow(l1_map, cmap="hot")
        plt.colorbar()
        plt.title("L1 distance heatmap")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    def save_histogram(self, values, out_path, title):
        plt.figure(figsize=(6, 5))
        plt.hist(values, bins=100, alpha=0.75)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    ################################################################
    # ANALYZE A SINGLE TRIPLET
    ################################################################

    def analyze_single(self, img_name):
        img_path = os.path.join(self.base_dir, img_name)
        student_path = os.path.join(self.student_dir, img_name.replace(".png", ".pt"))
        teacher_path = os.path.join(self.teacher_dir, img_name.replace(".png", ".pt"))

        # existence checks
        if not os.path.exists(student_path) or not os.path.exists(teacher_path):
            print(f"[SKIP] No matching .pt files for {img_name}")
            return None

        # load triplet - use concatenated image from base_dir
        img_concat = Image.open(img_path).convert("RGB")
        
        # Extract original image name (remove first number and underscore)
        # Example: "0_000000000139.png" -> "000000000139.png"
        original_img_name = (img_name.split("_", 1)[1] if "_" in img_name else img_name).replace(".png", ".jpg")
        # original_img_path = f"/scratch2/nico/distillation/dataset/coco2017/images/val2017/{original_img_name}"
        original_img_path = f"/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/distillation/dataset/coco2017/images/val2017/{original_img_name}"
        
        if not os.path.exists(original_img_path):
            print(f"[WARN] Original image not found: {original_img_path}")
            img = img_concat  # fallback to concatenated image
        else:
            img = Image.open(original_img_path).convert("RGB")
        
        student = self.load_embedding(student_path)
        teacher = self.load_embedding(teacher_path)

        # 1) L1 similarity
        l1_map, l1_mean = self.compute_l1(teacher, student)

        # 2) distributions
        t_mean, t_std, t_vals = self.compute_distribution(teacher)
        s_mean, s_std, s_vals = self.compute_distribution(student)

        # 3) save heatmap
        heatmap_path = os.path.join(
            self.out_dir, img_name.replace(".png", "_l1_heatmap.png")
        )
        self.save_heatmap(l1_map.numpy(), heatmap_path)

        # 3b) create overlay with threshold
        from scipy.ndimage import zoom
        threshold = 0.5
        l1_np = l1_map.numpy()
        
        # Resize heatmap to match image dimensions
        img_h, img_w = img.height, img.width
        heatmap_h, heatmap_w = l1_np.shape
        zoom_h = img_h / heatmap_h
        zoom_w = img_w / heatmap_w
        l1_resized = zoom(l1_np, (zoom_h, zoom_w), order=1)
        
        # Create mask for values > threshold
        mask_resized = l1_resized > threshold
        
        # Save overlay
        overlay_path = os.path.join(
            self.out_dir, img_name.replace(".png", f"_overlay_th{threshold}.png")
        )
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        
        # Overlay heatmap only where > threshold
        masked_heatmap = np.ma.masked_where(~mask_resized, l1_resized)
        im = ax.imshow(masked_heatmap, cmap="hot", alpha=0.6, interpolation="bilinear")
        
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.title(f"L1 Distance Overlay (threshold > {threshold})")
        plt.tight_layout()
        plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
        plt.close()

        # 4) save histograms
        hist_teacher_path = os.path.join(
            self.out_dir, img_name.replace(".png", "_teacher_hist.png")
        )
        hist_student_path = os.path.join(
            self.out_dir, img_name.replace(".png", "_student_hist.png")
        )

        self.save_histogram(t_vals, hist_teacher_path, "Teacher Feature Distribution")
        self.save_histogram(s_vals, hist_student_path, "Student Feature Distribution")

        # print summary
        print(f"\n===== {img_name} =====")
        print(f"L1 mean: {l1_mean:.4f}")
        print("Teacher mean/std: {:.4f} / {:.4f}".format(t_mean, t_std))
        print("Student mean/std: {:.4f} / {:.4f}".format(s_mean, s_std))
        print(f"Heatmap     → {heatmap_path}")
        print(f"Overlay     → {overlay_path}")
        print(f"Teacher hist→ {hist_teacher_path}")
        print(f"Student hist→ {hist_student_path}\n")

        return {
            "img": img_name,
            "l1_mean": l1_mean,
            "teacher_mean": t_mean,
            "teacher_std": t_std,
            "student_mean": s_mean,
            "student_std": s_std,
            "heatmap": heatmap_path,
            "overlay": overlay_path,
            "hist_teacher": hist_teacher_path,
            "hist_student": hist_student_path,
        }

    ################################################################
    # ANALYZE ALL
    ################################################################

    def analyze_all(self):
        print("\n=== Starting batch analysis ===")
        print(f"Base directory: {self.base_dir}\n")

        files = sorted([f for f in os.listdir(self.base_dir) if f.endswith(".png")])

        if not files:
            print("No .png files found in folder.")
            return

        results = []
        for img_name in files:
            res = self.analyze_single(img_name)
            if res is not None:
                results.append(res)

        print("\n=== Completed analysis for {} valid triplets ===".format(len(results)))
        print(f"Results saved in folder: {self.out_dir}\n")
        return results


################################################################
# CLI
################################################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_dir", type=str, default="/Users/nicoloiacobone/Desktop/nico/UNIVERSITA/MAGISTRALE/Tesi/Tommasi/Zurigo/git_clones/distillation/tests/SV_11_ALL_UNFROZEN/",
                        help="Folder containing images + student/teacher subfolders")
    args = parser.parse_args()

    analyzer = TripletAnalyzer(args.vis_dir)
    analyzer.analyze_all()