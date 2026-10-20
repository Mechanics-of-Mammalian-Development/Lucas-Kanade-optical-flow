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
