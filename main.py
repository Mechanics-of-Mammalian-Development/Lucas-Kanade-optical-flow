"""
Lucas-Kanade sparse optical flow inside the nucleus — movie + averaged map.

  1. make_movie()                : 3-panel movie (GIF + MP4 if ffmpeg is available)
                                   left   : native movie
                                   middle : feature points (Shi-Tomasi selection, then tracked)
                                   right  : Lucas-Kanade displacement vectors (coloured by speed)
  2. compute_and_plot_window()   : averaged speed heatmap + streamlines over a
                                   time window (SVG)

Stack: grayscale TIFF, shape (n_frames, H, W), uint8 or uint16
Switch each output on/off with RUN_MOVIE / RUN_MAP below.


REQUIREMENTS

numpy>=1.24
scipy>=1.10
matplotlib>=3.6        (>=3.6 needed for streamplot(broken_streamlines=...))
opencv-python>=4.8     (cv2: Shi-Tomasi + Lucas-Kanade)
scikit-image>=0.21     (io, measure, filters, morphology)
tifffile>=2023.1       (backend used by skimage.io to read the TIFF stack)
pillow>=9.0            (installed with matplotlib; writes the GIF via PillowWriter)
flexitext              (multicolour title)
(optional) ffmpeg      (system program, only for the MP4 output:
                        conda install -c conda-forge ffmpeg  or  brew install ffmpeg)

Install:
    pip install numpy scipy matplotlib opencv-python scikit-image tifffile pillow flexitext
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import shutil
import warnings
from collections import defaultdict


import numpy as np
from flexitext import flexitext
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import to_hex
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cv2
from skimage import io
from skimage.measure import label, regionprops
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, binary_erosion, disk, remove_small_objects
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
from scipy.interpolate import griddata

mpl.rcParams['svg.fonttype'] = 'path'   # text as paths: identical rendering everywhere
warnings.filterwarnings('ignore')


# =============================================================================
# 2. PARAMETERS
# =============================================================================
STACK_PATH         = 'Laurefull.tif'
DT                 = 0.5      # inter-frame interval (s)
PIXEL_UM           = 0.108    # µm per pixel

RUN_MOVIE          = True     # 3-panel movie
RUN_MAP            = True     # averaged flow-field map(s)

# ---- Nucleus detection ------------------------------------------------------
NUCLEUS_PERCENTILE = 50
NUCLEUS_MARGIN     = 2
NUCLEUS_SIGMA      = 0.9
NUCLEUS_CLOSE_R    = 8
NUCLEUS_MIN_AREA   = 0
NUCLEUS_USE_OTSU   = False

# ---- Background subtraction -------------------------------------------------
BG_SIGMA           = 10

# ---- Lucas-Kanade point detection (Shi-Tomasi) ------------------------------
LK_MAX_CORNERS     = 150
LK_QUALITY         = 0.01
LK_MIN_DIST        = 7
LK_BLOCK_SIZE      = 7

# ---- Lucas-Kanade tracking --------------------------------------------------
LK_WIN_SIZE        = (21, 21)
LK_MAX_LEVEL       = 3
LK_MAX_ITER        = 30
LK_EPSILON         = 0.01
REDETECT_EVERY     = 10

# ---- Movie ------------------------------------------------------------------
FRAME_START        = 0        # first frame
FRAME_END          = None     # last frame (None = end of stack)
FPS                = 5
ARROW_SCALE        = 3        # arrow length = displacement (px) x ARROW_SCALE
ARROW_WIDTH        = 0.007    # arrow shaft width (fraction of the axes width)
COLOR_NEW          = '#FF2A2A'  # points just detected (bright red)
COLOR_TRACKED      = '#9E1212'  # points carried over by tracking (dark red)
CONTRAST_PCT       = (1, 99.5)  # display range = these percentiles of the whole stack
                                # (narrower range = more contrast, e.g. (2, 99))
GAMMA              = 1.0      # <1 brightens dim structures, >1 darkens them
PANEL2_BACKGROUND  = 'native' # 'native' or 'prep' (background-subtracted image
                              # actually seen by the detector)
MOVIE_NAME         = 'lk_pipeline_movie'

# ---- Averaged map -----------------------------------------------------------
# Windows to plot: (label, first_frame, last_frame)
WINDOWS = [
    ('Ep1', 2, 21),
]
GRID               = 8        # grid spacing (px) for interpolation / streamlines
BIN_PX             = 12       # spatial bin (px) used to average vectors


# =============================================================================
# 3. LOAD STACK
# =============================================================================
stack = io.imread(STACK_PATH)
print(f"Stack: {stack.shape}, dtype: {stack.dtype}")
n_frames = stack.shape[0]
H, W = stack.shape[1:3]


# =============================================================================
# 4. SHARED FUNCTIONS: nucleus, preprocessing, Shi-Tomasi, LK parameters
# =============================================================================
def find_nucleus(frame, percentile=NUCLEUS_PERCENTILE, margin=NUCLEUS_MARGIN,
                 sigma=NUCLEUS_SIGMA, close_radius=NUCLEUS_CLOSE_R,
                 use_otsu=NUCLEUS_USE_OTSU, min_area=NUCLEUS_MIN_AREA):
    f = frame.astype(float)
    if sigma > 0:
        f = gaussian(f, sigma=sigma)
    thresh = threshold_otsu(f) if use_otsu else np.percentile(f, percentile)
    binary = f > thresh
    if close_radius > 0:
        binary = binary_closing(binary, disk(int(close_radius)))
    binary = ndi.binary_fill_holes(binary)
    if min_area > 0:
        binary = remove_small_objects(binary, int(min_area))
    labeled = label(binary)
    props = regionprops(labeled)
    if not props:
        return None
    largest = max(props, key=lambda r: r.area)
    cy, cx = largest.centroid
    radius = np.sqrt(largest.area / np.pi)
    nucleus_mask = labeled == largest.label
    if margin > 0:
        nucleus_mask = binary_erosion(nucleus_mask, disk(int(margin)))
    return cx, cy, radius, nucleus_mask


def prepare_frame(frame, sigma=BG_SIGMA):
    f = frame.astype(float)
    bg = gaussian(f, sigma=sigma)
    sub = np.clip(f - bg, 0, None)
    if sub.max() > 0:
        sub = sub / sub.max() * 255
    return sub.astype(np.uint8)


def detect_points(frame_uint8, nucleus_mask):
    mask_cv = nucleus_mask.astype(np.uint8) * 255
    pts = cv2.goodFeaturesToTrack(
        frame_uint8,
        maxCorners=LK_MAX_CORNERS,
        qualityLevel=LK_QUALITY,
        minDistance=LK_MIN_DIST,
        blockSize=LK_BLOCK_SIZE,
        mask=mask_cv
    )
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    return pts.astype(np.float32)


lk_params = dict(
    winSize=LK_WIN_SIZE,
    maxLevel=LK_MAX_LEVEL,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              LK_MAX_ITER, LK_EPSILON)
)

# Fallback mask if the nucleus is not found in some frame
_nuc0 = find_nucleus(stack[0])
mask0 = _nuc0[3] if _nuc0 else np.ones((H, W), dtype=bool)

EMPTY2 = np.empty((0, 2), dtype=np.float32)


# =============================================================================
# 5. THREE-PANEL MOVIE
# =============================================================================
def make_movie():
    f0 = FRAME_START
    f1 = n_frames - 1 if FRAME_END is None else min(FRAME_END, n_frames - 1)
    frames = list(range(f0, f1 + 1))
    print(f"\n--- Movie: frames {f0} -> {f1}  ({len(frames)} frames) ---")

    # ---- Run the pipeline once and record everything ----
    rec = {}   # frame -> dict

    nuc = find_nucleus(stack[f0])
    mask = nuc[3] if nuc else np.ones((H, W), dtype=bool)
    circle = (nuc[0], nuc[1], nuc[2]) if nuc else (W / 2, H / 2, min(H, W) / 2)

    prev = prepare_frame(stack[f0])
    pts  = detect_points(prev, mask)
    rec[f0] = dict(prep=prev, pts=pts.reshape(-1, 2), fresh=True,
                   vec_pts=EMPTY2, flows=EMPTY2, spd=np.empty(0), circle=circle)

    for fi in frames[1:]:
        curr = prepare_frame(stack[fi])
        nuc  = find_nucleus(stack[fi])
        if nuc:
            mask   = nuc[3]
            circle = (nuc[0], nuc[1], nuc[2])

        vec_pts, flows = EMPTY2, EMPTY2
        if len(pts) == 0:
            pts, fresh = detect_points(curr, mask), True
        else:
            pts_next, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts, None, **lk_params)
            ok      = status.ravel() == 1
            good_p  = pts[ok].reshape(-1, 2)
            good_n  = pts_next[ok].reshape(-1, 2)
            flows   = good_n - good_p
            vec_pts = good_n
            if fi % REDETECT_EVERY == 0:
                pts, fresh = detect_points(curr, mask), True
            else:
                pts, fresh = good_n.reshape(-1, 1, 2), False

        spd = (np.sqrt(flows[:, 0]**2 + flows[:, 1]**2) * PIXEL_UM / DT
               if len(flows) else np.empty(0))
        rec[fi] = dict(prep=curr, pts=pts.reshape(-1, 2), fresh=fresh,
                       vec_pts=vec_pts, flows=flows, spd=spd, circle=circle)
        prev = curr

    all_spd = np.concatenate([r['spd'] for r in rec.values() if len(r['spd'])]
                             or [np.array([0.1])])
    vmax_spd = max(np.percentile(all_spd, 95), 0.01)
    speed_norm = mcolors.Normalize(vmin=0, vmax=vmax_spd)
    speed_cmap = plt.get_cmap('inferno')
    print(f"95th percentile speed: {vmax_spd:.3f} µm/s")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.set_facecolor('black')
        ax.axis('off')
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)

    lo, hi = np.percentile(stack, CONTRAST_PCT)
    native_norm = mcolors.PowerNorm(gamma=GAMMA, vmin=lo, vmax=hi, clip=True)
    native_kw = dict(cmap='gray', norm=native_norm,
                     interpolation='nearest')   # raw pixels, no smoothing
    bg2_kw = (native_kw if PANEL2_BACKGROUND == 'native'
              else dict(cmap='gray', vmin=0, vmax=255, interpolation='nearest'))

    def bg2(fi):
        return stack[fi] if PANEL2_BACKGROUND == 'native' else rec[fi]['prep']

    im0 = axes[0].imshow(stack[f0], **native_kw)
    im1 = axes[1].imshow(bg2(f0),   **bg2_kw)
    im2 = axes[2].imshow(stack[f0], **native_kw)
    
    axes[0].set_title('H2B-mCherry', color='white', fontsize=22)

    flexitext(0.5, 1.02,
            '<size:22, color:white>'
            f'<color:{to_hex(COLOR_NEW)}>Detected</> and '
            f'<color:{to_hex(COLOR_TRACKED)}>tracked</> points'
            '</>',
            ax=axes[1], va='bottom', ha='center',
            xycoords='axes fraction')

    axes[2].set_title('Displacement vectors', color='white', fontsize=22)

    # Middle panel: nucleus circle + points
    circ = Circle(rec[f0]['circle'][:2], rec[f0]['circle'][2],
                  color='cyan', fill=False, lw=1.2, linestyle='--', alpha=0.6)
    axes[1].add_patch(circ)
    sc_old = axes[1].scatter([], [], s=18, c=COLOR_TRACKED, alpha=0.95, zorder=5)
    sc_new = axes[1].scatter([], [], s=28, c=COLOR_NEW,     alpha=0.95, zorder=6)


    # Right panel: arrows coloured by speed
    # (the quiver is re-created at every frame: a Quiver cannot change its
    #  number of arrows after creation)
    def make_quiver(vec_pts, flows, spd):
        if len(flows) == 0:
            vec_pts = np.array([[np.nan, np.nan]])
            flows   = np.zeros((1, 2))
            spd     = np.zeros(1)
        return axes[2].quiver(vec_pts[:, 0], vec_pts[:, 1], flows[:, 0], flows[:, 1], spd,
                              cmap=speed_cmap, norm=speed_norm,
                              angles='xy', scale_units='xy', scale=1.0 / ARROW_SCALE,
                              width=ARROW_WIDTH, headwidth=3.5, headlength=4, zorder=5)

    quiv = make_quiver(EMPTY2, EMPTY2, np.empty(0))

    # Inset colorbar (2 ticks) on the right panel
    cax = inset_axes(axes[2], width="30%", height="3%", loc='lower right', borderpad=0.5)
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), aspect='auto', cmap=speed_cmap,
               extent=[0, vmax_spd, 0, 1])
    cax.set_yticks([])
    cax.set_xticks([0, vmax_spd])
    cax.set_xticklabels(['0', f'{vmax_spd:.1f}'])
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.set_xlabel('Speed (µm/s)', color='white', fontsize=16)
    cax.tick_params(axis='x', colors='white', labelsize=12)
    for sp in cax.spines.values():
        sp.set_edgecolor('white')

    t_txt = fig.text(0.5, 0.01, '', ha='center', color='white', fontsize=22)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.05, wspace=0.02)

    # ---- Animation ----
    def update(k):
        nonlocal quiv
        fi = frames[k]
        r  = rec[fi]

        im0.set_array(stack[fi])
        im1.set_array(bg2(fi))
        im2.set_array(stack[fi])

        circ.center = r['circle'][:2]
        circ.radius = r['circle'][2]
        p = r['pts']
        if r['fresh']:
            sc_new.set_offsets(p if len(p) else EMPTY2)
            sc_old.set_offsets(EMPTY2)
        else:
            sc_old.set_offsets(p if len(p) else EMPTY2)
            sc_new.set_offsets(EMPTY2)


        quiv.remove()
        quiv = make_quiver(r['vec_pts'], r['flows'], r['spd'])

        t_txt.set_text(f"frame {fi}   t = {fi * DT:.1f} s")
        return [im0, im1, im2, circ, sc_old, sc_new, quiv, t_txt]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)

    # ---- Save ----
    print("Saving GIF...")
    anim.save(f"{MOVIE_NAME}.gif", writer=PillowWriter(fps=FPS))
    print(f"Saved: {MOVIE_NAME}.gif")

    if shutil.which('ffmpeg'):
        anim.save(f"{MOVIE_NAME}.mp4", writer=FFMpegWriter(fps=FPS, bitrate=8000))
        print(f"Saved: {MOVIE_NAME}.mp4")
    else:
        print("ffmpeg not found — MP4 skipped (conda install -c conda-forge ffmpeg)")

    plt.close(fig)


# =============================================================================
# 6. AVERAGED FLOW FIELD OVER A TIME WINDOW
# =============================================================================
def compute_and_plot_window(frame_start, frame_end, window_label='', fname=None):
    """Track points over [frame_start, frame_end], average the vectors on a
    grid, and plot speed heatmap + streamlines."""
    frames_sorted = list(range(frame_start, frame_end + 1))

    # --- Lucas-Kanade tracking over the window ---
    pos_flows = defaultdict(list)     # binned position -> list of flow vectors

    nuc_w  = find_nucleus(stack[frames_sorted[0]])
    mask_w = nuc_w[3] if nuc_w else mask0
    prev_w = prepare_frame(stack[frames_sorted[0]])
    pts_w  = detect_points(prev_w, mask_w)

    for fi in frames_sorted[1:]:
        curr_w = prepare_frame(stack[fi])
        if len(pts_w) == 0:
            nuc_w  = find_nucleus(stack[fi])
            mask_w = nuc_w[3] if nuc_w else mask0
            pts_w  = detect_points(curr_w, mask_w)
            prev_w = curr_w
            continue

        pts_next_w, status_w, _ = cv2.calcOpticalFlowPyrLK(
            prev_w, curr_w, pts_w, None, **lk_params)

        ok       = status_w.ravel() == 1
        good_p   = pts_w[ok]
        good_n   = pts_next_w[ok]
        flows_w  = good_n.reshape(-1, 2) - good_p.reshape(-1, 2)
        pts_2d_w = good_n.reshape(-1, 2)

        for pt, fl in zip(pts_2d_w, flows_w):
            key = (round(pt[0] / BIN_PX) * BIN_PX, round(pt[1] / BIN_PX) * BIN_PX)
            pos_flows[key].append(fl)

        if fi % REDETECT_EVERY == 0:
            nuc_w  = find_nucleus(stack[fi])
            mask_w = nuc_w[3] if nuc_w else mask0
            pts_w  = detect_points(curr_w, mask_w)
        else:
            pts_w = good_n.reshape(-1, 1, 2)
        prev_w = curr_w

    if not pos_flows:
        print(f"No data in window {frame_start}-{frame_end}")
        return

    # --- Regular grid for streamplot ---
    h_img, w_img = stack[0].shape
    gx = np.arange(GRID // 2, w_img, GRID)
    gy = np.arange(GRID // 2, h_img, GRID)
    GX, GY = np.meshgrid(gx, gy)

    SPD = np.full(GX.shape, np.nan)   # grid cells that contain a measurement
    for (bx, by), fls in pos_flows.items():
        mfl = np.mean(fls, axis=0)
        ix = np.argmin(np.abs(gx - bx))
        iy = np.argmin(np.abs(gy - by))
        SPD[iy, ix] = np.sqrt(mfl[0]**2 + mfl[1]**2) * PIXEL_UM / DT

    # Interpolate the sparse vectors onto the grid
    pts_known = np.array(list(pos_flows.keys()))
    u_known   = np.array([np.mean(f, axis=0)[0] for f in pos_flows.values()])
    v_known   = np.array([np.mean(f, axis=0)[1] for f in pos_flows.values()])

    U_interp = griddata(pts_known, u_known, (GX, GY), method='linear', fill_value=0)
    V_interp = griddata(pts_known, v_known, (GX, GY), method='linear', fill_value=0)
    SPD_interp = np.sqrt(U_interp**2 + V_interp**2) * PIXEL_UM / DT

    # Mask: blank out regions farther than 2 grid cells from any measurement
    has_data = ~np.isnan(SPD)
    mask_ok  = binary_dilation(has_data, iterations=2)
    U_interp[~mask_ok]   = np.nan
    V_interp[~mask_ok]   = np.nan
    SPD_interp[~mask_ok] = np.nan

    vmax_w  = np.nanpercentile(SPD_interp, 95)
    snorm_w = mcolors.Normalize(vmin=0, vmax=max(vmax_w, 0.01))

    # Average background image (kept as float: casting to uint8 would
    # overflow for uint16 stacks)
    bg_img = np.mean([stack[f] for f in frames_sorted if f < n_frames], axis=0)

    title = (f'{window_label}  frames {frame_start}–{frame_end}  '
             f'({frame_start * DT:.1f}s – {frame_end * DT:.1f}s)  '
             f'n={len(frames_sorted)} frames')

    # --- Plot ---
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    fig.patch.set_facecolor('black')
    fig.suptitle(title, color='white', fontsize=11)

    ax.set_facecolor('black')
    ax.imshow(bg_img, cmap='gray', vmin=stack.min(), vmax=stack.max(),
              alpha=0.5, rasterized=True)
    im_spd = ax.imshow(SPD_interp, extent=[0, w_img, h_img, 0],
                       cmap='inferno', norm=snorm_w, alpha=0.8,
                       interpolation='bilinear', rasterized=True)

    ax.streamplot(gx, gy, U_interp, V_interp,
                  color='white', linewidth=2.4, density=1,
                  arrowsize=1.5, broken_streamlines=True)

    # Inset colorbar (bottom right), 2 ticks, drawn by hand (SVG-safe)
    cax = inset_axes(ax, width="30%", height="3%", loc='lower right', borderpad=1.5)
    vmin, vmax = im_spd.get_clim()
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    cax.imshow(grad, aspect='auto', cmap=im_spd.cmap,
               extent=[vmin, vmax, 0, 1], alpha=1.0)
    cax.set_yticks([])
    cax.set_xticks([vmin, vmax])
    cax.set_xticklabels([f'{vmin:.1f}', f'{vmax:.1f}'])
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_label_position('top')
    cax.set_xlabel('Speed (µm/s)', color='white', fontsize=10)
    cax.tick_params(axis='x', colors='white', labelsize=10)
    for sp in cax.spines.values():
        sp.set_edgecolor('white')

    ax.set_title('Speed heatmap + field lines', color='white', fontsize=10)
    ax.axis('off')

    plt.tight_layout()
    fname = fname or f'flowfield_{window_label}_{frame_start}_{frame_end}'
    base = os.path.splitext(fname)[0]
    plt.savefig(base + '.svg', dpi=200, bbox_inches='tight', facecolor='black')
    plt.show()
    print(f"Saved: {base}.svg")


# =============================================================================
# 7. RUN
# =============================================================================
if __name__ == '__main__':
    if RUN_MOVIE:
        make_movie()
    if RUN_MAP:
        print("\n--- Averaged flow-field map(s) ---")
        for lbl, f_start, f_end in WINDOWS:
            compute_and_plot_window(f_start, f_end, window_label=lbl)
