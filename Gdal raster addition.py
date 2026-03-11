"""
raster_add.py
=============
Add two raster files (Band 1 of each) using GDAL — equivalent to the
QGIS Raster Calculator expression:  "raster1@1" + "raster2@1"

Pre-processing steps (automated):
  1. CRS check       – if CRS differ, raster 2 is reprojected to match raster 1.
  2. Pixel-size check – if pixel sizes differ, the coarser raster is resampled
                        (bilinear) to match the finer one.
  3. Grid alignment  – BOTH rasters are warped to a shared grid whose:
                         • origin is snapped to a common reference point so
                           pixels from both inputs line up exactly
                         • extent is the UNION of both rasters so no pixels
                           are lost at the edges

Null / NaN fill logic (per pixel):
  both valid          -> arr1 + arr2
  raster 1 null/NaN   -> use raster 2 value  (gap-fill from r2)
  raster 2 null/NaN   -> use raster 1 value  (gap-fill from r1)
  both null/NaN       -> output NoData

The output file is saved in the same directory as input raster 1.

Usage
-----
    python raster_add.py                         # interactive prompts
    python raster_add.py <input1> <input2> [output_name]

    # or import and call from another script:
    from raster_add import add_rasters
    add_rasters("dem.tif", "slope.tif", "result.tif")

Requirements
------------
    pip install gdal   # or conda install -c conda-forge gdal
"""

import math
import os
import sys
import tempfile

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_pixel_size(ds):
    """Return (x_res, y_res) as positive values."""
    gt = ds.GetGeoTransform()
    return abs(gt[1]), abs(gt[5])


def get_extent(ds):
    """Return (x_min, y_min, x_max, y_max) for the dataset."""
    gt = ds.GetGeoTransform()
    x_min = gt[0]
    y_max = gt[3]
    x_res = abs(gt[1])
    y_res = abs(gt[5])
    x_max = x_min + ds.RasterXSize * x_res
    y_min = y_max - ds.RasterYSize * y_res
    return x_min, y_min, x_max, y_max


def get_crs_wkt(ds):
    return ds.GetProjection()


def crs_are_equal(wkt1, wkt2):
    srs1, srs2 = osr.SpatialReference(), osr.SpatialReference()
    srs1.ImportFromWkt(wkt1)
    srs2.ImportFromWkt(wkt2)
    return bool(srs1.IsSame(srs2))


def reproject_raster(src_path, target_wkt, tmp_dir, tag="reprojected"):
    """Reproject src_path to target_wkt; return path to a temp GeoTIFF."""
    out_path = os.path.join(tmp_dir, f"{tag}.tif")
    gdal.Warp(
        out_path,
        src_path,
        dstSRS=target_wkt,
        resampleAlg=gdal.GRA_Bilinear,
        format="GTiff",
    )
    print(f"  [CRS] Reprojected '{os.path.basename(src_path)}' -> target CRS")
    return out_path


def resample_raster(src_path, target_x_res, target_y_res, tmp_dir, tag="resampled"):
    """Resample src_path to (target_x_res, target_y_res); return temp path."""
    out_path = os.path.join(tmp_dir, f"{tag}.tif")
    gdal.Warp(
        out_path,
        src_path,
        xRes=target_x_res,
        yRes=target_y_res,
        resampleAlg=gdal.GRA_Bilinear,
        format="GTiff",
    )
    src_res = get_pixel_size(gdal.Open(src_path))
    print(f"  [Pixel] Resampled '{os.path.basename(src_path)}' "
          f"from {src_res[0]:.6g} -> {target_x_res:.6g}")
    return out_path


def snap_value(value, origin, res, direction="floor"):
    """
    Snap `value` to the nearest grid line defined by `origin` and `res`.
    direction='floor' -> snap outward toward smaller values  (use for x_min / y_min)
    direction='ceil'  -> snap outward toward larger values   (use for x_max / y_max)
    """
    offset = (value - origin) / res
    if direction == "floor":
        return origin + math.floor(offset) * res
    else:
        return origin + math.ceil(offset) * res


def build_common_grid(ds1, ds2, x_res, y_res):
    """
    Compute a shared grid that:
      - covers the UNION extent of both datasets
      - has pixel size (x_res, y_res)
      - is snapped so that pixel edges from BOTH original grids align
        (we use the top-left corner of ds1 as the grid anchor)

    Returns (x_min, y_min, x_max, y_max, cols, rows).
    """
    e1 = get_extent(ds1)
    e2 = get_extent(ds2)

    # Union extent (raw)
    ux_min = min(e1[0], e2[0])
    uy_min = min(e1[1], e2[1])
    ux_max = max(e1[2], e2[2])
    uy_max = max(e1[3], e2[3])

    # Anchor: top-left of ds1 — both rasters were already on the same CRS
    # and (after resampling) the same pixel size, so ds1's origin is the
    # natural snapping reference.
    ax = e1[0]   # anchor x (left edge of ds1)
    ay = e1[3]   # anchor y (top  edge of ds1)

    # Snap union extent outward to the nearest grid line from the anchor
    snapped_x_min = snap_value(ux_min, ax, x_res, "floor")
    snapped_x_max = snap_value(ux_max, ax, x_res, "ceil")
    snapped_y_min = snap_value(uy_min, ay, -y_res, "floor")   # y axis is inverted
    snapped_y_max = snap_value(uy_max, ay, -y_res, "ceil")

    cols = int(round((snapped_x_max - snapped_x_min) / x_res))
    rows = int(round((snapped_y_max - snapped_y_min) / y_res))

    return snapped_x_min, snapped_y_min, snapped_x_max, snapped_y_max, cols, rows


def warp_to_grid(src_path, x_min, y_min, x_max, y_max, x_res, y_res,
                 target_wkt, tmp_dir, tag):
    """
    Warp src_path to an exact grid (extent + pixel size + CRS).
    NoData areas introduced by the warp are set to NaN so our
    null-fill logic handles them correctly.
    """
    out_path = os.path.join(tmp_dir, f"{tag}.tif")
    gdal.Warp(
        out_path,
        src_path,
        dstSRS=target_wkt,
        outputBounds=(x_min, y_min, x_max, y_max),
        xRes=x_res,
        yRes=y_res,
        resampleAlg=gdal.GRA_Bilinear,
        srcNodata=None,          # preserve source nodata flag
        dstNodata=float("nan"),  # use NaN for pixels outside source coverage
        format="GTiff",
        creationOptions=["COMPRESS=LZW"],
    )
    return out_path


def read_band(path, band=1):
    """Return (array float64, nodata_value) for the given band."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    b = ds.GetRasterBand(band)
    nodata = b.GetNoDataValue()
    arr = b.ReadAsArray().astype(np.float64)
    return arr, nodata


def invalid_mask(arr, nodata):
    """
    True wherever a pixel is invalid:
      - IEEE NaN
      - matches the raster's registered NoData value (float tolerance)
    """
    mask = np.isnan(arr)
    if nodata is not None and not math.isnan(nodata):
        mask |= np.isclose(arr, nodata)
    return mask


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def add_rasters(
    path1,
    path2,
    out_path,
    band1=1,
    band2=1,
    out_nodata=-9999.0,
):
    """
    Add raster1 + raster2 pixel-by-pixel and write the result to out_path.

    Where one raster has a null/NaN pixel but the other is valid, the valid
    pixel value is used directly (gap-fill) instead of propagating NoData.
    Only pixels that are null/NaN in *both* inputs become NoData in the output.
    """
    print("=" * 60)
    print(f"Input 1 : {path1}")
    print(f"Input 2 : {path2}")
    print(f"Output  : {out_path}")
    print("=" * 60)

    ds1 = gdal.Open(path1, gdal.GA_ReadOnly)
    ds2 = gdal.Open(path2, gdal.GA_ReadOnly)

    if ds1 is None:
        raise FileNotFoundError(f"Cannot open: {path1}")
    if ds2 is None:
        raise FileNotFoundError(f"Cannot open: {path2}")

    with tempfile.TemporaryDirectory() as tmp_dir:

        # ── 1. CRS harmonisation ──────────────────────────────────────────
        wkt1 = get_crs_wkt(ds1)
        wkt2 = get_crs_wkt(ds2)

        if not wkt1:
            print("  [WARN] Raster 1 has no CRS defined — proceeding anyway.")
        if not wkt2:
            print("  [WARN] Raster 2 has no CRS defined — proceeding anyway.")

        work_path1 = path1
        work_path2 = path2

        if wkt1 and wkt2 and not crs_are_equal(wkt1, wkt2):
            print("  [CRS] Mismatch — reprojecting raster 2 to match raster 1.")
            work_path2 = reproject_raster(work_path2, wkt1, tmp_dir, tag="r2_reproj")
        else:
            print("  [CRS] CRS match ✓")

        # Reload ds1/ds2 from working paths (ds1 may still be original)
        wd1 = gdal.Open(work_path1, gdal.GA_ReadOnly)
        wd2 = gdal.Open(work_path2, gdal.GA_ReadOnly)

        # ── 2. Pixel-size harmonisation ───────────────────────────────────
        x1, y1 = get_pixel_size(wd1)
        x2, y2 = get_pixel_size(wd2)

        print(f"  [Pixel] Raster 1 pixel size: {x1:.6g} x {y1:.6g}")
        print(f"  [Pixel] Raster 2 pixel size: {x2:.6g} x {y2:.6g}")

        tol = 1e-9
        if abs(x1 - x2) > tol or abs(y1 - y2) > tol:
            fine_x = min(x1, x2)
            fine_y = min(y1, y2)
            if x1 > fine_x or y1 > fine_y:
                print("  [Pixel] Raster 1 is coarser — resampling raster 1.")
                work_path1 = resample_raster(work_path1, fine_x, fine_y, tmp_dir, tag="r1_resamp")
            else:
                print("  [Pixel] Raster 2 is coarser — resampling raster 2.")
                work_path2 = resample_raster(work_path2, fine_x, fine_y, tmp_dir, tag="r2_resamp")
            x_res, y_res = fine_x, fine_y
        else:
            x_res, y_res = x1, y1
            print("  [Pixel] Pixel sizes match ✓")

        # Reload after potential resampling
        wd1 = gdal.Open(work_path1, gdal.GA_ReadOnly)
        wd2 = gdal.Open(work_path2, gdal.GA_ReadOnly)

        # ── 3. Build common union grid (snapped to ds1 anchor) ────────────
        x_min, y_min, x_max, y_max, out_cols, out_rows = build_common_grid(
            wd1, wd2, x_res, y_res
        )
        print(f"  [Grid] Union extent  : ({x_min:.4f}, {y_min:.4f}) -> ({x_max:.4f}, {y_max:.4f})")
        print(f"  [Grid] Output size   : {out_cols} cols x {out_rows} rows")

        target_wkt = wkt1 if wkt1 else ""

        # Warp BOTH rasters to the identical grid
        aligned1 = warp_to_grid(
            work_path1, x_min, y_min, x_max, y_max, x_res, y_res,
            target_wkt, tmp_dir, tag="r1_aligned"
        )
        aligned2 = warp_to_grid(
            work_path2, x_min, y_min, x_max, y_max, x_res, y_res,
            target_wkt, tmp_dir, tag="r2_aligned"
        )
        print("  [Grid] Both rasters warped to common grid ✓")

        # ── 4. Read arrays ────────────────────────────────────────────────
        arr1, nd1 = read_band(aligned1, band1)
        arr2, nd2 = read_band(aligned2, band2)

        # Sanity check — shapes must match after grid alignment
        if arr1.shape != arr2.shape:
            raise RuntimeError(
                f"Shape mismatch after alignment: {arr1.shape} vs {arr2.shape}. "
                "This should not happen — please report as a bug."
            )

        # ── 5. Build per-pixel null / NaN masks ───────────────────────────
        # warp_to_grid uses dstNodata=NaN, so out-of-coverage areas are NaN.
        # invalid_mask catches both NaN and any explicit NoData value.
        null1 = invalid_mask(arr1, nd1)
        null2 = invalid_mask(arr2, nd2)

        both_null  = null1 & null2
        only_null1 = null1 & ~null2   # r1 missing, r2 has data
        only_null2 = ~null1 & null2   # r2 missing, r1 has data

        print(f"  [NoData] Raster 1 null/NaN pixels : {int(null1.sum()):,}")
        print(f"  [NoData] Raster 2 null/NaN pixels : {int(null2.sum()):,}")

        # ── 6. Fill nulls BEFORE adding ───────────────────────────────────
        #
        # Each raster's null pixels are replaced with the other raster's
        # value so that the addition in step 7 always operates on two
        # fully-populated arrays (no null can contaminate the sum).
        #
        #   r1 null, r2 valid  ->  fill r1 with r2 value
        #   r2 null, r1 valid  ->  fill r2 with r1 value
        #   both null          ->  leave as-is (handled after addition)
        #
        filled1 = arr1.copy()
        filled1[only_null1] = arr2[only_null1]   # patch r1 gaps from r2

        filled2 = arr2.copy()
        filled2[only_null2] = arr1[only_null2]   # patch r2 gaps from r1

        print(f"  [Fill] Pixels in r1 filled from r2              : {int(only_null1.sum()):,}")
        print(f"  [Fill] Pixels in r2 filled from r1              : {int(only_null2.sum()):,}")
        print(f"  [Fill] Pixels null in both (-> NoData after add) : {int(both_null.sum()):,}")

        # ── 7. Add the filled arrays ──────────────────────────────────────
        #
        # Both arrays are now null-free except where BOTH were originally
        # null. Those positions are set to out_nodata in the output.
        #
        result = filled1 + filled2
        result[both_null] = out_nodata

        print(f"  [Add]  Valid output pixels : {int((~both_null).sum()):,}")

        # ── 8. Write output ───────────────────────────────────────────────
        driver = gdal.GetDriverByName("GTiff")
        out_ds = driver.Create(
            out_path,
            out_cols,
            out_rows,
            1,
            gdal.GDT_Float64,
            options=["COMPRESS=LZW", "TILED=YES"],
        )
        # GeoTransform: top-left corner + pixel size
        out_gt = (x_min, x_res, 0.0, y_max, 0.0, -y_res)
        out_ds.SetGeoTransform(out_gt)
        out_ds.SetProjection(target_wkt)

        out_band = out_ds.GetRasterBand(1)
        out_band.SetNoDataValue(out_nodata)
        out_band.WriteArray(result)
        out_band.FlushCache()
        out_ds.FlushCache()
        out_ds = None

        print(f"\n  ✓ Output written : {out_path}")
        print(f"    Shape          : {out_rows} rows x {out_cols} cols")
        valid_pixels = result[~both_null]
        if valid_pixels.size:
            print(f"    Min            : {valid_pixels.min():.6g}")
            print(f"    Max            : {valid_pixels.max():.6g}")
            print(f"    Mean           : {valid_pixels.mean():.6g}")
        print("=" * 60)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def prompt_input_file(label):
    """Prompt the user for a raster file path, validating it exists."""
    while True:
        raw = input(f"{label}: ").strip().strip('"').strip("'")
        if not raw:
            print("  [Error] Path cannot be empty. Please try again.")
            continue
        path = os.path.abspath(raw)
        if not os.path.isfile(path):
            print(f"  [Error] File not found: {path}")
            print("  Please check the path and try again.")
            continue
        return path


def build_output_path(input1, output_name=None):
    """
    Derive the output path:
      - Always placed in the same directory as input1.
      - If output_name given, use only its basename (.tif appended if needed).
      - Otherwise default to 'output_sum.tif'.
    """
    out_dir = os.path.dirname(os.path.abspath(input1))
    if not output_name:
        output_name = "output_sum.tif"
    output_name = os.path.basename(output_name)
    if not os.path.splitext(output_name)[1]:
        output_name += ".tif"
    return os.path.join(out_dir, output_name)


def prompt_output_name(input1):
    """Ask the user for an output filename (just the name, no path needed)."""
    out_dir = os.path.dirname(os.path.abspath(input1))
    print(f"\n  Output will be saved to: {out_dir}")
    raw = input("  Output filename (press Enter for default 'output_sum.tif'): ").strip().strip('"').strip("'")
    return build_output_path(input1, raw if raw else None)


# ---------------------------------------------------------------------------
# CLI / interactive entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("  Raster Addition Tool  (GDAL-based QGIS Raster Calculator)")
    print("=" * 60)

    if len(sys.argv) >= 3:
        in1 = os.path.abspath(sys.argv[1])
        in2 = os.path.abspath(sys.argv[2])
        if not os.path.isfile(in1):
            print(f"[Error] Input 1 not found: {in1}")
            sys.exit(1)
        if not os.path.isfile(in2):
            print(f"[Error] Input 2 not found: {in2}")
            sys.exit(1)
        out_name = sys.argv[3] if len(sys.argv) >= 4 else None
        out = build_output_path(in1, out_name)
    else:
        print("\nEnter the full path to each raster file.")
        print("(Tip: you can drag-and-drop a file into the terminal)\n")
        in1 = prompt_input_file("Input raster 1")
        in2 = prompt_input_file("Input raster 2")
        out = prompt_output_name(in1)

    print(f"\n  Input 1 : {in1}")
    print(f"  Input 2 : {in2}")
    print(f"  Output  : {out}\n")

    if os.path.abspath(out) in (os.path.abspath(in1), os.path.abspath(in2)):
        print("[Error] Output path must differ from both input paths.")
        sys.exit(1)

    add_rasters(in1, in2, out)
