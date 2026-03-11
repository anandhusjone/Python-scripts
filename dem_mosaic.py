"""
dem_mosaic.py
=============
Mosaic a large collection of DEM raster files into a single GeoTIFF.
Optimised for huge datasets: no full-array loads, block-wise processing,
GDAL native operations throughout.

Pipeline:
  1. INPUT       – folder or explicit file list; auto-discovers rasters.
  2. CRS CHECK   – inspect every file's CRS via metadata only (no pixel reads).
                   Reproject non-conforming files with gdal.Warp streaming.
  3. NODATA FILL – run gdal.FillNodata() per file in streaming/tiled mode,
                   iterating with a growing search radius until no NoData
                   remains or the iteration cap is hit.
                   NoData counting is done block-by-block (constant memory).
  4. MOSAIC      – build a VRT first, then stream-warp to the final GeoTIFF.
                   No tile is ever fully loaded into RAM.
  5. OUTPUT      – LZW-compressed, tiled GeoTIFF; output stats computed
                   block-by-block.

Memory footprint at any time:
  ≈ max(BLOCK_SIZE_PX² × 8 bytes × 2)  ≈ a few MB regardless of DEM count
  (controlled by BLOCK_SIZE_PX and GDAL_CACHE_MB below)

Usage
-----
    python dem_mosaic.py                    # fully interactive
    python dem_mosaic.py /path/to/dem/dir   # directory of DEMs
    python dem_mosaic.py a.tif b.tif c.tif  # explicit file list

Requirements
------------
    pip install gdal   # or  conda install -c conda-forge gdal
"""

import math
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter

from osgeo import gdal, osr

gdal.UseExceptions()

# ── tunables ──────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS  = {".tif", ".tiff", ".img", ".hgt", ".vrt", ".adf", ".dem"}
ZIP_DEM_SUFFIX        = ".dem.tif" # suffix to match inside zips (e.g. "*.dem.tif"); case-insensitive
ZIP_DEM_FILENAME      = "dem.tif"  # legacy exact-basename fallback (used if suffix match finds nothing)
ZIP_SCAN_ALL_RASTERS  = True       # last-resort fallback: grab any supported raster in the zip
ZIP_LIST_ON_SKIP      = True       # print zip contents when no match found (helps diagnose names)
GDAL_CACHE_MB         = 512    # GDAL block cache — raise if you have more RAM
BLOCK_SIZE_PX         = 512    # tile size for output GeoTIFF and processing
FILL_INITIAL_RADIUS   = 10     # pixels — starting FillNodata search radius
FILL_RADIUS_STEP      = 20     # pixels — added to radius each iteration
FILL_MAX_ITERATIONS   = 5      # hard stop per file
FILL_SMOOTH_ITER      = 0      # smoothing passes inside FillNodata (0 = none)
WARP_MEMORY_MB        = 512    # memory budget passed to gdal.Warp
# ─────────────────────────────────────────────────────────────────────────────

# Apply GDAL cache setting immediately
gdal.SetCacheMax(GDAL_CACHE_MB * 1024 * 1024)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def extract_from_zips(folder, extract_dir):
    """
    Scan `folder` for .zip files and extract DEM rasters from each one.

    Matching strategy (in order):
      1. Any member whose filename ends with ZIP_DEM_SUFFIX (e.g. "*.dem.tif").
      2. Any member whose exact basename matches ZIP_DEM_FILENAME (legacy fallback).
      3. If ZIP_SCAN_ALL_RASTERS=True and still no match, grab any file whose
         extension is in SUPPORTED_EXTENSIONS.

    Each zip extracts into its own subdirectory so files never overwrite each other.
    If ZIP_LIST_ON_SKIP=True, the zip contents are printed when nothing matches
    (helps diagnose unexpected internal file names).

    Returns a list of extracted file paths.
    """
    extracted = []
    zip_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(".zip")
    )
    if not zip_files:
        return extracted

    print(f"  [ZIP] Found {len(zip_files)} zip file(s) in {folder}")

    for zname in zip_files:
        zpath    = os.path.join(folder, zname)
        zip_stem = os.path.splitext(zname)[0]

        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                all_members = [
                    m for m in zf.namelist()
                    if not os.path.basename(m).startswith("._")   # skip macOS meta
                    and not m.endswith("/")                         # skip directories
                ]

                # Strategy 1: suffix match — e.g. "*.dem.tif"
                matches = [
                    m for m in all_members
                    if os.path.basename(m).lower().endswith(ZIP_DEM_SUFFIX.lower())
                ]

                # Strategy 2: exact basename fallback
                if not matches:
                    matches = [
                        m for m in all_members
                        if os.path.basename(m).lower() == ZIP_DEM_FILENAME.lower()
                    ]
                    if matches:
                        names = ", ".join(os.path.basename(m) for m in matches[:5])
                        print(f"    {zname}: suffix match — {names}")

                # Strategy 3: last resort — any supported raster extension
                if not matches and ZIP_SCAN_ALL_RASTERS:
                    matches = [
                        m for m in all_members
                        if os.path.splitext(m)[1].lower() in SUPPORTED_EXTENSIONS
                    ]
                    if matches:
                        names = ", ".join(os.path.basename(m) for m in matches[:5])
                        print(f"    {zname}: no suffix match — "
                              f"using fallback raster(s): {names}")

                if not matches:
                    print(f"    {zname}: no raster found, skipping")
                    if ZIP_LIST_ON_SKIP:
                        raster_like = [m for m in all_members if "." in os.path.basename(m)][:10]
                        if raster_like:
                            print(f"      Files inside: {', '.join(os.path.basename(m) for m in raster_like)}")
                    continue

                for member in matches:
                    safe_name = member.replace("/", "_").replace("\\", "_")
                    out_path  = os.path.join(extract_dir, zip_stem, safe_name)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with zf.open(member) as src, open(out_path, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"    {zname}/{member} → extracted")
                    extracted.append(out_path)

        except zipfile.BadZipFile:
            print(f"    [WARN] {zname}: bad/incomplete zip file, skipping")
        except Exception as e:
            print(f"    [WARN] {zname}: error ({e}), skipping")

    print(f"  [ZIP] Extracted {len(extracted)} file(s) from zips")
    return extracted


def list_zip_contents(path):
    """Print all members inside a zip file — used for diagnostics."""
    import zipfile as _zf
    try:
        with _zf.ZipFile(path, "r") as zf:
            members = zf.namelist()
        print(f"  Contents of {os.path.basename(path)} ({len(members)} entries):")
        for m in members[:40]:
            print(f"    {m}")
        if len(members) > 40:
            print(f"    ... and {len(members)-40} more")
    except _zf.BadZipFile:
        print(f"  [WARN] {os.path.basename(path)}: bad zip file")


def discover_rasters(paths, extract_dir=None):
    """
    Expand folders, zip files, and explicit file paths into a flat sorted
    list of raster paths.

    For each folder encountered:
      • Regular raster files with SUPPORTED_EXTENSIONS are collected directly.
      • Any .zip files in the folder are scanned for ZIP_DEM_FILENAME and
        the matching files are extracted to `extract_dir` (a temp directory
        managed by the caller) before being added to the list.

    For explicit file paths:
      • If the path is a .zip file it is treated the same as a zip in a folder.
      • Otherwise the file is added directly.
    """
    found = []

    for p in paths:
        p = os.path.abspath(p)

        if os.path.isdir(p):
            # Collect regular rasters
            for fname in sorted(os.listdir(p)):
                if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTENSIONS:
                    found.append(os.path.join(p, fname))
            # Scan zips in the same folder
            if extract_dir:
                found.extend(extract_from_zips(p, extract_dir))

        elif os.path.isfile(p):
            if p.lower().endswith(".zip"):
                # Explicit zip file — treat its parent dir as the scan folder
                folder = os.path.dirname(p)
                if extract_dir:
                    try:
                        with zipfile.ZipFile(p, "r") as zf:
                            matches = [
                                m for m in zf.namelist()
                                if os.path.basename(m).lower().endswith(ZIP_DEM_SUFFIX.lower())
                                and not os.path.basename(m).startswith("._")
                            ]
                            if not matches:
                                print(f"  [ZIP] {os.path.basename(p)}: "
                                      f"no \'{ZIP_DEM_FILENAME}\' found, skipping")
                            for member in matches:
                                zip_stem  = os.path.splitext(os.path.basename(p))[0]
                                safe_name = member.replace("/", "_").replace("\\", "_")
                                out_path  = os.path.join(extract_dir, zip_stem, safe_name)
                                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                                with zf.open(member) as src, open(out_path, "wb") as dst:
                                    shutil.copyfileobj(src, dst)
                                print(f"  [ZIP] {os.path.basename(p)}/{member} → extracted")
                                found.append(out_path)
                    except zipfile.BadZipFile:
                        print(f"  [WARN] {os.path.basename(p)}: bad zip, skipping")
            else:
                found.append(p)

        else:
            print(f"  [WARN] Path not found, skipping: {p}")

    return found


def get_crs_wkt(path):
    """Read CRS from file metadata only — no pixel data loaded."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    wkt = ds.GetProjection()
    ds = None
    return wkt or ""


def get_epsg_label(wkt):
    """Return 'AUTHORITY:CODE' string or short name for display."""
    if not wkt:
        return "UNKNOWN"
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    srs.AutoIdentifyEPSG()
    code = srs.GetAuthorityCode(None)
    name = srs.GetAuthorityName(None)
    if code and name:
        return f"{name}:{code}"
    return srs.GetName() or wkt[:60]


def crs_are_equal(wkt1, wkt2):
    srs1, srs2 = osr.SpatialReference(), osr.SpatialReference()
    srs1.ImportFromWkt(wkt1)
    srs2.ImportFromWkt(wkt2)
    return bool(srs1.IsSame(srs2))


def get_file_info(path):
    """Return dict of file metadata without reading pixel data."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    info = {
        "cols":    ds.RasterXSize,
        "rows":    ds.RasterYSize,
        "nodata":  band.GetNoDataValue(),
        "dtype":   band.DataType,
        "wkt":     ds.GetProjection() or "",
        "gt":      ds.GetGeoTransform(),
    }
    ds = None
    return info


# ---------------------------------------------------------------------------
# Block-wise NoData counter  (constant memory)
# ---------------------------------------------------------------------------

def count_nodata_blocks(path, band_idx=1):
    """
    Count NoData / NaN pixels by reading one block at a time.
    Peak RAM = one block of float64 ≈ BLOCK_SIZE_PX² × 8 bytes.
    Returns (nodata_value, nodata_count, total_pixels).
    """
    ds   = gdal.Open(path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(band_idx)
    nd   = band.GetNoDataValue()
    cols = ds.RasterXSize
    rows = ds.RasterYSize

    bx, by = band.GetBlockSize()          # natural tile size from the file
    bx = max(bx, BLOCK_SIZE_PX)          # at least our minimum block size
    by = max(by, BLOCK_SIZE_PX)

    nd_count = 0
    total    = 0

    for y0 in range(0, rows, by):
        ysize = min(by, rows - y0)
        for x0 in range(0, cols, bx):
            xsize = min(bx, cols - x0)
            chunk = band.ReadAsArray(x0, y0, xsize, ysize).astype("float64")
            mask  = __import__("numpy").isnan(chunk)
            if nd is not None and not (isinstance(nd, float) and math.isnan(nd)):
                mask |= __import__("numpy").isclose(chunk, nd)
            nd_count += int(mask.sum())
            total    += chunk.size
            chunk = None   # free immediately

    ds = None
    return nd, nd_count, total


# ---------------------------------------------------------------------------
# Step 2 – CRS harmonisation (streaming, no pixel reads)
# ---------------------------------------------------------------------------

def harmonise_crs(file_list, tmp_dir):
    """
    Reproject any file whose CRS differs from the majority CRS.
    Uses gdal.Warp in streaming mode — never loads a full raster into RAM.
    Returns (working_path_list, target_wkt).
    """
    print("\n── Step 2: CRS check ──────────────────────────────────────────")

    wkts   = [get_crs_wkt(f) for f in file_list]
    labels = [get_epsg_label(w) for w in wkts]

    for f, lbl in zip(file_list, labels):
        print(f"  {os.path.basename(f):45s}  {lbl}")

    non_empty = [(w, l) for w, l in zip(wkts, labels) if w]
    if not non_empty:
        print("  [WARN] No CRS found in any file — skipping reprojection.")
        return file_list, ""

    # Majority vote on CRS
    label_counts = Counter(l for _, l in non_empty)
    target_label, _ = label_counts.most_common(1)[0]
    target_wkt = next(w for w, l in non_empty if l == target_label)
    print(f"\n  Target CRS : {target_label}")

    work_paths   = []
    reproj_count = 0

    for i, (f, wkt) in enumerate(zip(file_list, wkts)):
        if wkt and crs_are_equal(wkt, target_wkt):
            work_paths.append(f)
            continue

        out = os.path.join(tmp_dir, f"reproj_{i:04d}.tif")
        print(f"  Reprojecting: {os.path.basename(f)} …")

        gdal.Warp(
            out, f,
            dstSRS=target_wkt,
            resampleAlg=gdal.GRA_Bilinear,
            format="GTiff",
            warpMemoryLimit=WARP_MEMORY_MB * 1024 * 1024,
            creationOptions=[
                f"BLOCKXSIZE={BLOCK_SIZE_PX}",
                f"BLOCKYSIZE={BLOCK_SIZE_PX}",
                "TILED=YES",
                "COMPRESS=LZW",
            ],
        )
        work_paths.append(out)
        reproj_count += 1

    status = f"Reprojected {reproj_count} file(s) ✓" if reproj_count else "All files share the same CRS ✓"
    print(f"  {status}")
    return work_paths, target_wkt


# ---------------------------------------------------------------------------
# Step 3 – Iterative NoData fill  (block-wise counting, streaming fill)
# ---------------------------------------------------------------------------

def ensure_nodata_tag(path, nd):
    """
    Make sure the file has a NoData value registered so FillNodata can
    locate holes.  Writes the tag in-place — no pixel data copied.
    Returns the (possibly updated) nodata value.
    """
    import numpy as np
    ds   = gdal.Open(path, gdal.GA_Update)
    band = ds.GetRasterBand(1)

    if nd is None:
        dtype = band.DataType
        nd = float("nan") if dtype in (gdal.GDT_Float32, gdal.GDT_Float64) else -9999.0
        band.SetNoDataValue(nd)

        # Sweep the file block-by-block: replace any stray NaN with nd
        cols, rows = ds.RasterXSize, ds.RasterYSize
        bx, by = BLOCK_SIZE_PX, BLOCK_SIZE_PX
        for y0 in range(0, rows, by):
            ysize = min(by, rows - y0)
            for x0 in range(0, cols, bx):
                xsize = min(bx, cols - x0)
                chunk = band.ReadAsArray(x0, y0, xsize, ysize).astype("float64")
                nan_m = np.isnan(chunk)
                if nan_m.any():
                    chunk[nan_m] = nd
                    band.WriteArray(chunk, x0, y0)
                chunk = None
        ds.FlushCache()

    ds = None
    return nd


def fill_nodata_iterative(src_path, tmp_dir, index):
    """
    Iteratively fill NoData holes in src_path using gdal.FillNodata().
    Works on a tiled copy so the original is never modified.
    NoData counting is block-wise → constant memory.
    Returns path to filled file (original if no holes found).
    """
    nd, nd_count, total = count_nodata_blocks(src_path)

    label = os.path.basename(src_path)
    if nd_count == 0:
        print(f"  [{index:3d}] {label:45s}  no NoData ✓")
        return src_path

    pct = 100.0 * nd_count / total
    print(f"  [{index:3d}] {label:45s}  {nd_count:,} NoData px ({pct:.2f}%)")

    # Copy to a tiled working file (enables efficient block access)
    work = os.path.join(tmp_dir, f"filled_{index:04d}.tif")
    gdal.Translate(
        work, src_path,
        format="GTiff",
        creationOptions=[
            f"BLOCKXSIZE={BLOCK_SIZE_PX}",
            f"BLOCKYSIZE={BLOCK_SIZE_PX}",
            "TILED=YES",
            "COMPRESS=LZW",
        ],
    )

    nd = ensure_nodata_tag(work, nd)

    radius    = FILL_INITIAL_RADIUS
    iteration = 0

    while nd_count > 0 and iteration < FILL_MAX_ITERATIONS:
        iteration += 1

        ds   = gdal.Open(work, gdal.GA_Update)
        band = ds.GetRasterBand(1)
        mask = band.GetMaskBand()

        gdal.FillNodata(
            targetBand=band,
            maskBand=mask,
            maxSearchDist=radius,
            smoothingIterations=FILL_SMOOTH_ITER,
        )
        ds.FlushCache()
        ds = None   # close before re-counting

        _, nd_count, _ = count_nodata_blocks(work)
        print(f"        iter {iteration:2d}  radius={radius:5d} px  "
              f"remaining={nd_count:,}")
        radius += FILL_RADIUS_STEP

    outcome = "fully filled ✓" if nd_count == 0 else f"{nd_count:,} px remain (gap too large)"
    print(f"        → {outcome} after {iteration} iteration(s)")
    return work


# ---------------------------------------------------------------------------
# Step 4 – Mosaic via VRT then stream-warp  (no full raster in RAM)
# ---------------------------------------------------------------------------

def mosaic_rasters(file_list, out_path, target_wkt, tmp_dir):
    """
    1. Build an in-memory VRT that stitches all files together.
    2. Stream-warp the VRT to the final GeoTIFF.
    No tile is ever fully loaded into RAM.
    """
    print("\n── Step 4: Mosaicing ──────────────────────────────────────────")
    print(f"  Building VRT from {len(file_list)} file(s) …")

    vrt_path = os.path.join(tmp_dir, "mosaic.vrt")
    vrt_opts = gdal.BuildVRTOptions(
        resampleAlg="bilinear",
        srcNodata=-9999,
        VRTNodata=-9999,
        addAlpha=False,
    )
    vrt_ds = gdal.BuildVRT(vrt_path, file_list, options=vrt_opts)
    vrt_ds.FlushCache()
    vrt_ds = None
    print("  VRT built ✓")

    print(f"  Streaming warp to: {out_path}")
    gdal.Warp(
        out_path,
        vrt_path,
        dstSRS=target_wkt if target_wkt else None,
        resampleAlg=gdal.GRA_Bilinear,
        srcNodata=-9999,
        dstNodata=-9999,
        format="GTiff",
        warpMemoryLimit=WARP_MEMORY_MB * 1024 * 1024,
        multithread=True,
        creationOptions=[
            f"BLOCKXSIZE={BLOCK_SIZE_PX}",
            f"BLOCKYSIZE={BLOCK_SIZE_PX}",
            "TILED=YES",
            "COMPRESS=LZW",
            "BIGTIFF=IF_SAFER",
        ],
        callback=gdal.TermProgress_nocb,
    )
    print()   # newline after progress bar


# ---------------------------------------------------------------------------
# Step 5 – Output stats  (block-wise, constant memory)
# ---------------------------------------------------------------------------

def print_output_stats(out_path):
    """Compute min/max/mean block-by-block without loading the full raster."""
    import numpy as np

    print("\n── Output statistics (computed block-wise) ────────────────────")
    ds   = gdal.Open(out_path, gdal.GA_ReadOnly)
    band = ds.GetRasterBand(1)
    nd   = band.GetNoDataValue()
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    gt   = ds.GetGeoTransform()

    bx = by = BLOCK_SIZE_PX
    nd_total = 0
    total    = 0
    vmin     =  math.inf
    vmax     = -math.inf
    vsum     = 0.0
    vcnt     = 0

    for y0 in range(0, rows, by):
        ysize = min(by, rows - y0)
        for x0 in range(0, cols, bx):
            xsize = min(bx, cols - x0)
            chunk = band.ReadAsArray(x0, y0, xsize, ysize).astype("float64")
            mask  = np.isnan(chunk)
            if nd is not None and not math.isnan(nd):
                mask |= np.isclose(chunk, nd)
            valid = chunk[~mask]
            nd_total += int(mask.sum())
            total    += chunk.size
            if valid.size:
                vmin  = min(vmin, float(valid.min()))
                vmax  = max(vmax, float(valid.max()))
                vsum += float(valid.sum())
                vcnt += valid.size
            chunk = valid = None

    ds = None
    x_res = abs(gt[1])
    y_res = abs(gt[5])

    print(f"  File             : {out_path}")
    print(f"  Size             : {rows} rows x {cols} cols")
    print(f"  Pixel size       : {x_res:.6g} x {y_res:.6g}")
    print(f"  NoData pixels    : {nd_total:,} / {total:,} "
          f"({100.0*nd_total/total:.3f}%)")
    if vcnt:
        print(f"  Elevation min    : {vmin:.4g}")
        print(f"  Elevation max    : {vmax:.4g}")
        print(f"  Elevation mean   : {vsum/vcnt:.4g}")
    if nd_total:
        print(f"  [WARN] {nd_total:,} NoData pixels remain "
              "(holes too large to fill from neighbours)")
    else:
        print("  No NoData pixels in mosaic ✓")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def prompt_file_list():
    print("\nEnter a folder path  OR  individual raster file paths, one per line.")
    print("Press Enter on an empty line when done.\n")
    entries = []
    while True:
        raw = input("  Path (or Enter to finish): ").strip().strip('"').strip("'")
        if not raw:
            if entries:
                break
            print("  [Error] Please enter at least one path.")
            continue
        entries.append(raw)
    return entries  # raw paths; discovery + zip extraction done in mosaic_dems


def prompt_output_path(file_list):
    out_dir = os.path.dirname(os.path.abspath(file_list[0]))
    print(f"\n  Output will be saved to: {out_dir}")
    raw  = input("  Output filename [default: mosaic_dem.tif]: ").strip().strip('"').strip("'")
    name = raw if raw else "mosaic_dem.tif"
    name = os.path.basename(name)
    if not os.path.splitext(name)[1]:
        name += ".tif"
    return os.path.join(out_dir, name)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def mosaic_dems(raw_paths, out_path):
    print("=" * 65)
    print("  DEM Mosaic Tool  —  memory-optimised")
    print("=" * 65)
    print(f"  GDAL cache       : {GDAL_CACHE_MB} MB")
    print(f"  Warp memory      : {WARP_MEMORY_MB} MB")
    print(f"  Block size       : {BLOCK_SIZE_PX} px")
    print(f"  ZIP target suffix: {ZIP_DEM_SUFFIX}  (fallback: {ZIP_DEM_FILENAME})")
    print(f"  Output           : {out_path}")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Step 1 – Discover rasters (including zip extraction)
        zip_extract_dir = os.path.join(tmp_dir, "zip_extracted")
        os.makedirs(zip_extract_dir, exist_ok=True)
        print("\n── Step 1: File discovery ─────────────────────────────────────")
        file_list = discover_rasters(raw_paths, extract_dir=zip_extract_dir)
        if not file_list:
            raise ValueError("No raster files found after discovery.")
        print(f"  Total rasters found : {len(file_list)}")
        for f in file_list:
            info = get_file_info(f)
            size_mpx = info["cols"] * info["rows"] / 1e6
            print(f"    {os.path.basename(f):45s}  "
                  f"{info['cols']}x{info['rows']} ({size_mpx:.1f} Mpx)")

        # Step 2 – CRS
        work_paths, target_wkt = harmonise_crs(file_list, tmp_dir)

        # Step 3 – Fill NoData per file
        print("\n── Step 3: NoData fill ────────────────────────────────────────")
        filled_paths = []
        for i, path in enumerate(work_paths, start=1):
            filled_paths.append(fill_nodata_iterative(path, tmp_dir, i))

        # Step 4 – Mosaic
        mosaic_rasters(filled_paths, out_path, target_wkt, tmp_dir)

    # Step 5 – Stats (after tmp_dir closed — reads from final output only)
    print_output_stats(out_path)

    print(f"\n  ✓ Done: {out_path}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  DEM Mosaic Tool  (GDAL-based, memory-optimised)")
    print("=" * 65)

    if len(sys.argv) > 1:
        raw_paths = [os.path.abspath(p) for p in sys.argv[1:]]
    else:
        raw_paths = [os.path.abspath(p) for p in prompt_file_list()]

    if not raw_paths:
        print("[Error] No paths provided.")
        sys.exit(1)

    # Determine output path using the first non-zip path as anchor,
    # or fall back to the first path's parent directory.
    anchor = next(
        (p for p in raw_paths if not p.lower().endswith(".zip") and os.path.isfile(p)),
        raw_paths[0]
    )
    if len(sys.argv) > 1:
        out_dir  = os.path.dirname(anchor) if os.path.isfile(anchor) else anchor
        out_path = os.path.join(out_dir, "mosaic_dem.tif")
        print(f"  Output : {out_path}")
    else:
        # For interactive mode, prompt with a dummy list for directory display
        out_path = prompt_output_path([anchor])

    mosaic_dems(raw_paths, out_path)