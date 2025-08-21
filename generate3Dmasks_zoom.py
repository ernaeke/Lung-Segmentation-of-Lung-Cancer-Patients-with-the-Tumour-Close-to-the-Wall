import os
import csv
import json
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from numpy.linalg import inv
from unet3d import UNet3D

# =====================
# Config
# =====================

ROI_AS_3CM_CUBE = False      # True => 3D cube; False => thin slab (cz-1..cz+1)
PRED_THRESH = 0.2
SAVE_CONTEXT_IMAGE = True

# Auto-pick tries these viewer sizes for display coords (W,H)
CANDIDATE_DISPLAY_SIZES = [(256, 256), (512, 512), (462, 493)]

# Reuse the same display->array mapping for all ROIs in a case
STICKY_MAPPING = True

# Persist mapping across runs (so nudges never flip)
PERSIST_MAPPING = True
MAPPING_DB_FILENAME = "display_mapping.json"

# NEW: Use one global default mapping for every case (prevents cross-case flips)
USE_GLOBAL_DEFAULT_MAPPING = True
# Set this to the mapping that worked for your first case
GLOBAL_DEFAULT_MAPPING = {"disp": (512, 512), "swap": True, "fx": True, "fy": False}

# =====================
# Utilities
# =====================

def dice_score(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return 2.0 * inter / (pred.sum() + gt.sum() + 1e-8)

def get_spacing_mm(img_nii):
    sx, sy, sz = map(float, img_nii.header.get_zooms()[:3])
    return sx, sy, sz

def load_tumor_centers(csv_path):
    """
    CSV columns (flexible):
      Case,x,y,z,box_half,coords_space[,dispW,dispH,map_swap,map_flipX,map_flipY,dx,dy]
    - coords_space: display | voxel | world | mm
    - dispW/H (optional): viewer pixel size for that row
    - map_* (optional): 0/1 to force mapping for that row/case
    - dx,dy (optional): display-space nudges (+x right, +y down)
    """
    centers = {}
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for ln, row in enumerate(reader, start=2):
            if not row:
                continue
            case = (row.get("Case") or "").strip()
            if not case or case.startswith("#"):
                continue
            try:
                cx = float((row.get("x") or "").strip())
                cy = float((row.get("y") or "").strip())
                cz = float((row.get("z") or "").strip())
                bh = int((row.get("box_half") or "0").strip())
                space = (row.get("coords_space") or "display").strip().lower()

                dispW = row.get("dispW"); dispH = row.get("dispH")
                dispW = int(float(dispW)) if dispW not in (None, "") else None
                dispH = int(float(dispH)) if dispH not in (None, "") else None

                ms  = row.get("map_swap");   ms  = None if ms  in (None, "") else bool(int(ms))
                mfx = row.get("map_flipX");  mfx = None if mfx in (None, "") else bool(int(mfx))
                mfy = row.get("map_flipY");  mfy = None if mfy in (None, "") else bool(int(mfy))

                dx  = float(row.get("dx") or 0.0)
                dy  = float(row.get("dy") or 0.0)
            except Exception as e:
                print(f"↷ Skipping malformed row (line {ln}): {e} -> {row}")
                continue

            centers.setdefault(case, []).append({
                "cx": cx, "cy": cy, "cz": cz, "box_half": bh, "space": space,
                "dispW": dispW, "dispH": dispH,
                "map_swap": ms, "map_fx": mfx, "map_fy": mfy,
                "dx": dx, "dy": dy
            })
    return centers

def roi_bounds_3cm_square(cx, cy, cz, spacing_mm, shape_xyz, as_cube=False):
    sx, sy, sz = spacing_mm
    half_mm = 15.0
    hx = int(round(half_mm / max(sx, 1e-6)))
    hy = int(round(half_mm / max(sy, 1e-6)))
    x0, x1 = int(cx) - hx, int(cx) + hx
    y0, y1 = int(cy) - hy, int(cy) + hy
    if as_cube:
        hz = int(round(half_mm / max(sz, 1e-6)))
        z0, z1 = int(cz) - hz, int(cz) + hz
    else:
        z0, z1 = int(cz) - 1, int(cz) + 1

    X, Y, Z = shape_xyz
    x0, y0, z0 = max(0, x0), max(0, y0), max(0, z0)
    x1, y1, z1 = min(X, x1), min(Y, y1), min(Z, z1)
    return x0, x1, y0, y1, z0, z1

# ---------- viewer orientation helpers ----------

def _transform_image_for_display(img2d, params):
    fx = params.get("fx", False); fy = params.get("fy", False); swap = params.get("swap", False)
    if fx:   img2d = np.fliplr(img2d)
    if fy:   img2d = np.flipud(img2d)
    if swap: img2d = img2d.T
    return img2d

def _to_display_coords(row, col, X, Y, params):
    fx = params.get("fx", False); fy = params.get("fy", False); swap = params.get("swap", False)
    if fx:   col = (Y - 1) - col
    if fy:   row = (X - 1) - row
    if swap: row, col = col, row; X, Y = Y, X
    return row, col, X, Y

def _rect_to_display(x0, x1, y0, y1, X, Y, params):
    pts = [(x0, y0), (x0, y1), (x1, y1), (x1, y0)]
    trans = [_to_display_coords(r, c, X, Y, params)[:2] for (r, c) in pts]
    rs = [r for r, _ in trans]; cs = [c for _, c in trans]
    return min(rs), max(rs), min(cs), max(cs)

# ---------- display→array mapping & selection ----------

def _apply_display_mapping_to_rc(cx, cy, X, Y, dispW, dispH, swap, fx, fy):
    dx, dy = (cy, cx) if swap else (cx, cy)
    col_f = dx * (Y / max(dispW, 1))
    row_f = dy * (X / max(dispH, 1))
    if fx: col_f = (Y - 1) - col_f
    if fy: row_f = (X - 1) - row_f
    row = int(round(max(0, min(row_f, X - 1))))
    col = int(round(max(0, min(col_f, Y - 1))))
    return row, col

def _roi_bounds_from_center(row, col, slc, spacing_mm, shape_xyz, as_cube=False):
    x0, x1, y0, y1, z0, z1 = roi_bounds_3cm_square(row, col, slc, spacing_mm, shape_xyz, as_cube=as_cube)
    X, Y, Z = shape_xyz
    x0 = max(0, min(x0, X - 1)); x1 = max(0, min(x1, X))
    y0 = max(0, min(y0, Y - 1)); y1 = max(0, min(y1, Y))
    z0 = max(0, min(z0, Z - 1)); z1 = max(0, min(z1, Z))
    return x0, x1, y0, y1, z0, z1

def auto_select_display_mapping(center, image_np, gt_mask_np, spacing_mm):
    """
    center: {... 'cx','cy','cz','dispW','dispH','dx','dy'}
    Returns (params,row,col,slc) where params={'disp':(W,H),'swap','fx','fy'}
    Picks mapping whose 3 cm ROI overlaps lung mask (or non-air) the most.
    """
    X, Y, Z = image_np.shape
    cx = center["cx"] + float(center.get("dx", 0.0))
    cy = center["cy"] + float(center.get("dy", 0.0))
    slc = max(0, min(int(center["cz"]), Z - 1))
    use_lung = gt_mask_np is not None and gt_mask_np.size > 0

    sizes = []
    if center.get("dispW") and center.get("dispH"):
        sizes.append((int(center["dispW"]), int(center["dispH"])))
    else:
        sizes = list(CANDIDATE_DISPLAY_SIZES) + [(Y, X)]

    best = None
    for dispW, dispH in sizes:
        for swap in (False, True):
            for fx in (False, True):
                for fy in (False, True):
                    row, col = _apply_display_mapping_to_rc(cx, cy, X, Y, dispW, dispH, swap, fx, fy)
                    x0, x1, y0, y1, _, _ = _roi_bounds_from_center(row, col, slc, spacing_mm, image_np.shape, False)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    if use_lung:
                        score = gt_mask_np[x0:x1, y0:y1, slc].sum()
                    else:
                        patch = image_np[x0:x1, y0:y1, slc]
                        score = (patch > (200/1400.0)).sum()
                    params = {'disp': (dispW, dispH), 'swap': swap, 'fx': fx, 'fy': fy}
                    if (best is None) or (score > best[0]):
                        best = (score, params, row, col)
        if center.get("dispW") and center.get("dispH"):
            break

    if best is None:
        params = {'disp': (Y, X), 'swap': False, 'fx': False, 'fy': False}
        return params, X//2, Y//2, slc

    _, params, row, col = best
    return params, row, col, slc

def map_display_fixed(center, image_shape, params):
    """Map display coords to array (row,col,slc) using a fixed mapping params."""
    X, Y, Z = image_shape
    cx = center["cx"] + float(center.get("dx", 0.0))
    cy = center["cy"] + float(center.get("dy", 0.0))
    dispW, dispH = params['disp']
    row, col = _apply_display_mapping_to_rc(cx, cy, X, Y, dispW, dispH,
                                            params['swap'], params['fx'], params['fy'])
    slc = max(0, min(int(center["cz"]), Z - 1))
    return row, col, slc

# ---------- mapping DB (persist across runs) ----------

def _as_tuple_wh(x):
    return tuple(x) if isinstance(x, (list, tuple)) else (int(x), int(x))

def load_mapping_db(path):
    try:
        with open(path, "r") as f:
            db = json.load(f)
        for k, v in db.items():
            v["disp"] = _as_tuple_wh(v["disp"])
            v["swap"] = bool(v.get("swap", False))
            v["fx"]   = bool(v.get("fx", False))
            v["fy"]   = bool(v.get("fy", False))
        return db
    except Exception:
        return {}

def save_mapping_db(path, db):
    safe = {k: {"disp": list(v["disp"]), "swap": bool(v["swap"]),
                "fx": bool(v["fx"]), "fy": bool(v["fy"])} for k, v in db.items()}
    with open(path, "w") as f:
        json.dump(safe, f, indent=2)

# =====================
# Visualization
# =====================

def save_roi_visual(case_dir, case_name, roi_id, ct_crop, gt_crop, pred_crop, spacing_mm, viewer_params=None):
    """Save zoomed-in ROI visuals with physical size annotation."""
    if ct_crop.size == 0:
        print(f"⚠️ Empty crop for {case_name} ROI {roi_id}, skipping visualization.")
        return
    mid = ct_crop.shape[2] // 2
    img = ct_crop[:, :, mid]; gt = gt_crop[:, :, mid]; pr = pred_crop[:, :, mid]
    if viewer_params is not None:
        img = _transform_image_for_display(img, viewer_params)
        gt  = _transform_image_for_display(gt,  viewer_params)
        pr  = _transform_image_for_display(pr,  viewer_params)
    sx, sy, _ = spacing_mm
    size_x_mm = img.shape[0] * sx; size_y_mm = img.shape[1] * sy

    fig, axs = plt.subplots(1, 3, figsize=(9, 3), dpi=150)
    axs[0].imshow(img, cmap="gray", interpolation="nearest"); axs[0].set_title(f"CT (~{size_x_mm:.1f}×{size_y_mm:.1f} mm)"); axs[0].axis("off")
    axs[1].imshow(img, cmap="gray", interpolation="nearest"); axs[1].imshow(gt, cmap="Reds", alpha=0.4, interpolation="nearest"); axs[1].set_title("GT");   axs[1].axis("off")
    axs[2].imshow(img, cmap="gray", interpolation="nearest"); axs[2].imshow(pr, cmap="Greens", alpha=0.4, interpolation="nearest"); axs[2].set_title("Pred"); axs[2].axis("off")
    plt.tight_layout()
    out_dir = os.path.join(case_dir, "roi_visuals"); os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{case_name}_roi{roi_id}.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0); plt.close()
    print(f"🖼 Saved zoom-in ROI visual: {out_path}")

def save_context_visual(case_dir, case_name, roi_id, full_slice, rect_bounds, center_rc, z_idx,
                        viewer_params=None, label=None, subdir="roi_context"):
    X, Y = full_slice.shape
    x0, x1, y0, y1 = rect_bounds; r, c = center_rc
    if viewer_params is not None:
        disp = _transform_image_for_display(full_slice, viewer_params)
        r_d, c_d, _, _ = _to_display_coords(r, c, X, Y, viewer_params)
        x0_d, x1_d, y0_d, y1_d = _rect_to_display(x0, x1, y0, y1, X, Y, viewer_params)
    else:
        disp = full_slice; r_d, c_d = r, c; x0_d, x1_d, y0_d, y1_d = x0, x1, y0, y1
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=150)
    ax.imshow(disp, cmap='gray', interpolation='nearest')
    ax.plot([y0_d, y1_d, y1_d, y0_d, y0_d], [x0_d, x0_d, x1_d, x1_d, x0_d], '-', lw=1.5, color='red')
    ax.plot([c_d-5, c_d+5], [r_d, r_d], '-', color='yellow', lw=1.2)
    ax.plot([c_d, c_d], [r_d-5, r_d+5], '-', color='yellow', lw=1.2)
    ax.set_title(label or f"Slice {z_idx} (context)"); ax.axis('off')
    out_dir = os.path.join(case_dir, subdir); os.makedirs(out_dir, exist_ok=True)
    name = f"{case_name}_roi{roi_id}.png" if not label else f"{case_name}_roi{roi_id}_{label}.png"
    out_path = os.path.join(out_dir, name)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0); plt.close()
    print(f"🗺 Saved context: {out_path}")

# =====================
# Core
# =====================

def generate_mask_3d(model, scan_path, mask_dir, output_root, csv_global, csv_regions,
                     tumor_centers, device, mapping_cache, mapping_db_path):
    case_name = os.path.splitext(os.path.basename(scan_path))[0]
    print(f"Processing: {case_name}")

    # Load scan
    img_nii = nib.load(scan_path)
    spacing_mm = get_spacing_mm(img_nii)
    affine = img_nii.affine
    image_np = np.clip(img_nii.get_fdata(), 0, 1400) / 1400.0

    # Load GT mask
    mask_path = os.path.join(mask_dir, f"{case_name}.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(mask_dir, f"{case_name}.nii")
    if not os.path.exists(mask_path):
        print(f"⚠️ GT mask missing for {case_name}")
        return

    gt_mask_np = (nib.load(mask_path).get_fdata() > 0).astype(np.uint8)

    # Predict
    image_tensor = torch.tensor(image_np.transpose(2, 1, 0)[None, None], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(image_tensor)
        pred_soft = pred.squeeze().detach().cpu().numpy()
        pred_mask = (pred_soft > PRED_THRESH).astype(np.uint8)
    pred_mask = pred_mask.transpose(2, 1, 0)

    # Align Z
    minZ = min(pred_mask.shape[2], image_np.shape[2], gt_mask_np.shape[2])
    image_np, gt_mask_np, pred_mask = image_np[:, :, :minZ], gt_mask_np[:, :, :minZ], pred_mask[:, :, :minZ]

    # Global Dice
    csv_global.writerow([case_name, f"{dice_score(pred_mask, gt_mask_np):.4f}"])

    # Per-ROI
    if case_name in tumor_centers:
        for i, center in enumerate(tumor_centers[case_name], start=1):
            viewer_params = None

            if center["space"] == "display":
                # 1) Forced mapping via CSV? (always obey)
                if all(k in center and center[k] is not None for k in ("map_swap","map_fx","map_fy","dispW","dispH")):
                    params = {
                        "disp": (int(center["dispW"]), int(center["dispH"])),
                        "swap": bool(center["map_swap"]),
                        "fx":   bool(center["map_fx"]),
                        "fy":   bool(center["map_fy"]),
                    }
                    cx_idx, cy_idx, cz_idx = map_display_fixed(center, image_np.shape, params)
                    viewer_params = {"disp": params["disp"], "swap": params["swap"], "fx": params["fx"], "fy": params["fy"]}
                    print(f"[{case_name} ROI {i}] forced W{params['disp'][0]}x{params['disp'][1]} "
                          f"swap={params['swap']} fx={params['fx']} fy={params['fy']} "
                          f"-> array (row,col,slc)=({cx_idx},{cy_idx},{cz_idx})")
                    # remember per-case and persist
                    mapping_cache[case_name] = params
                    if PERSIST_MAPPING:
                        save_mapping_db(mapping_db_path, mapping_cache)

                # 2) Case-specific persisted mapping?
                elif STICKY_MAPPING and (case_name in mapping_cache):
                    params = mapping_cache[case_name]
                    cx_idx, cy_idx, cz_idx = map_display_fixed(center, image_np.shape, params)
                    viewer_params = {"disp": params["disp"], "swap": params["swap"], "fx": params["fx"], "fy": params["fy"]}
                    print(f"[{case_name} ROI {i}] using persisted W{params['disp'][0]}x{params['disp'][1]} "
                          f"swap={params['swap']} fx={params['fx']} fy={params['fy']} "
                          f"-> array (row,col,slc)=({cx_idx},{cy_idx},{cz_idx})")

                # 3) Global default mapping (prevents flips across cases)
                elif USE_GLOBAL_DEFAULT_MAPPING and ("__global__" in mapping_cache):
                    params = mapping_cache["__global__"]
                    cx_idx, cy_idx, cz_idx = map_display_fixed(center, image_np.shape, params)
                    viewer_params = {"disp": params["disp"], "swap": params["swap"], "fx": params["fx"], "fy": params["fy"]}
                    print(f"[{case_name} ROI {i}] using GLOBAL W{params['disp'][0]}x{params['disp'][1]} "
                          f"swap={params['swap']} fx={params['fx']} fy={params['fy']} "
                          f"-> array (row,col,slc)=({cx_idx},{cy_idx},{cz_idx})")
                    # also store per-case and persist for traceability
                    mapping_cache[case_name] = params
                    if PERSIST_MAPPING:
                        save_mapping_db(mapping_db_path, mapping_cache)

                # 4) Last resort: auto-pick; also establish global if none yet
                else:
                    params, cx_idx, cy_idx, cz_idx = auto_select_display_mapping(center, image_np, gt_mask_np, spacing_mm)
                    viewer_params = {"disp": params["disp"], "swap": params["swap"], "fx": params["fx"], "fy": params["fy"]}
                    mapping_cache[case_name] = params
                    if USE_GLOBAL_DEFAULT_MAPPING and "__global__" not in mapping_cache:
                        mapping_cache["__global__"] = params
                    if PERSIST_MAPPING:
                        save_mapping_db(mapping_db_path, mapping_cache)
                    print(f"[{case_name} ROI {i}] picked W{params['disp'][0]}x{params['disp'][1]} "
                          f"swap={params['swap']} fx={params['fx']} fy={params['fy']} "
                          f"-> array (row,col,slc)=({cx_idx},{cy_idx},{cz_idx})")

            elif center["space"] == "voxel":
                X, Y, Z = image_np.shape
                cx_idx = max(0, min(int(center["cx"]), X - 1))
                cy_idx = max(0, min(int(center["cy"]), Y - 1))
                cz_idx = max(0, min(int(center["cz"]), Z - 1))

            else:  # 'world' / 'mm'
                Ainv = inv(affine)
                i_, j_, k_, _ = Ainv @ np.array([center["cx"], center["cy"], center["cz"], 1.0], dtype=np.float64)
                X, Y, Z = image_np.shape
                cx_idx = max(0, min(int(round(i_)), X - 1))
                cy_idx = max(0, min(int(round(j_)), Y - 1))
                cz_idx = max(0, min(int(round(k_)), Z - 1))

            # ROI bounds (3 cm)
            x0, x1, y0, y1, z0, z1 = roi_bounds_3cm_square(
                cx_idx, cy_idx, cz_idx, spacing_mm, image_np.shape, as_cube=ROI_AS_3CM_CUBE
            )
            x0 = max(0, min(x0, image_np.shape[0]-1)); x1 = max(0, min(x1, image_np.shape[0]))
            y0 = max(0, min(y0, image_np.shape[1]-1)); y1 = max(0, min(y1, image_np.shape[1]))
            z0 = max(0, min(z0, image_np.shape[2]-1)); z1 = max(0, min(z1, image_np.shape[2]))

            ct_crop   = image_np[x0:x1, y0:y1, z0:z1]
            gt_crop   = gt_mask_np[x0:x1, y0:y1, z0:z1]
            pred_crop = pred_mask[x0:x1, y0:y1, z0:z1]
            if ct_crop.size == 0 or gt_crop.size == 0 or pred_crop.size == 0:
                print(f"⚠️ Empty ROI for {case_name} ROI {i} — skipping.")
                continue

            dice_roi = dice_score(pred_crop, gt_crop)
            roi_type = "3cm_cube" if ROI_AS_3CM_CUBE else "3cm_square"

            # CSV row (include mapping if display-space)
            if viewer_params is not None:
                dispW, dispH = viewer_params["disp"]
                csv_regions.writerow([case_name, i, cx_idx, cy_idx, cz_idx, center["space"],
                                      roi_type, f"{dice_roi:.4f}", dispW, dispH,
                                      int(viewer_params["swap"]), int(viewer_params["fx"]), int(viewer_params["fy"])])
            else:
                csv_regions.writerow([case_name, i, cx_idx, cy_idx, cz_idx, center["space"],
                                      roi_type, f"{dice_roi:.4f}", "", "", "", "", ""])

            # Save images
            case_dir = os.path.join(output_root, case_name); os.makedirs(case_dir, exist_ok=True)
            save_roi_visual(case_dir, case_name, i, ct_crop, gt_crop, pred_crop, spacing_mm, viewer_params)
            if SAVE_CONTEXT_IMAGE:
                z_mid = cz_idx if not ROI_AS_3CM_CUBE else (z0 + z1)//2
                full_slice = image_np[:, :, z_mid]
                save_context_visual(case_dir, case_name, i, full_slice,
                                    rect_bounds=(x0, x1, y0, y1),
                                    center_rc=(cx_idx, cy_idx), z_idx=z_mid,
                                    viewer_params=viewer_params)

# =====================
# Script entry
# =====================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet3D().to(device)
    model.load_state_dict(torch.load("path_to.pth", map_location=device))
    model.eval()

    input_dir   = "Path_to_scans"
    mask_dir    = "Path_to_masks"
    centers_csv = "Path_to_coordinates_of_tumours.csv"
    output_root = "zoomGenwithaug25epochs"

    os.makedirs(output_root, exist_ok=True)

    # mapping DB (persist across runs)
    mapping_db_path = os.path.join(output_root, MAPPING_DB_FILENAME)
    mapping_cache   = load_mapping_db(mapping_db_path)

    # NEW: seed a global mapping so all cases share the same orientation
    if USE_GLOBAL_DEFAULT_MAPPING and "__global__" not in mapping_cache:
        mapping_cache["__global__"] = GLOBAL_DEFAULT_MAPPING
        if PERSIST_MAPPING:
            save_mapping_db(mapping_db_path, mapping_cache)

    tumor_centers = load_tumor_centers(centers_csv)

    with open(os.path.join(output_root, "dice_global.csv"), "w", newline="") as f_global, \
         open(os.path.join(output_root, "dice_regions.csv"), "w", newline="") as f_regions:

        csv_g = csv.writer(f_global); csv_g.writerow(["Case", "Dice_Global"])
        csv_r = csv.writer(f_regions)
        csv_r.writerow([
            "Case","CenterID","cx_idx","cy_idx","cz_idx","coords_space",
            "roi_type","Dice_Region","map_dispW","map_dispH","map_swap","map_flipX","map_flipY"
        ])

        for fname in sorted(os.listdir(input_dir)):
            if fname.endswith(".nii") or fname.endswith(".nii.gz"):
                scan_path = os.path.join(input_dir, fname)
                generate_mask_3d(model, scan_path, mask_dir, output_root, csv_g, csv_r,
                                 tumor_centers, device, mapping_cache, mapping_db_path)

    print(f"✅ Done. Results saved in: {output_root}")
