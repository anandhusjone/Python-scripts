import os
import csv
import time
import sys
import subprocess
import requests
from datetime import datetime, timedelta

#makesure fmpeg is installed in your system for video creation
#find the start and end code from "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/GEOCOLOR/"

def parse_code(code):
    year = int(code[:4])
    julian = int(code[4:7])
    hour = int(code[7:9])
    minute = int(code[9:11])
    return datetime(year, 1, 1, hour, minute) + timedelta(days=julian - 1)

def generate_filenames(start_code, end_code, prefix, csv_path):
    start_dt = parse_code(start_code)
    end_dt = parse_code(end_code)

    names = []
    dt = start_dt
    while dt <= end_dt:
        year = dt.year
        julian = dt.timetuple().tm_yday
        hhmm = f"{dt.hour:02d}{dt.minute:02d}"
        names.append(f"{year}{julian:03d}{hhmm}{prefix}")
        dt += timedelta(minutes=5)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        for name in names:
            writer.writerow([name])

    print(f"Generated {len(names)} filenames → {csv_path}")
    return names

def download_images(filenames, save_dir, base_url):
    os.makedirs(save_dir, exist_ok=True)
    total = len(filenames)
    durations = []

    for i, fname in enumerate(filenames, 1):
        url = base_url + fname
        dest = os.path.join(save_dir, fname)
        t0 = time.time()
        try:
            r = requests.get(url, stream=True, timeout=20)
            if r.status_code == 200:
                with open(dest, "wb") as out:
                    for chunk in r.iter_content(8192):
                        out.write(chunk)
                status = "OK"
            else:
                status = "Missing"
        except Exception as e:
            status = f"Err:{e}"

        t1 = time.time()
        durations.append(t1 - t0)
        avg = sum(durations) / len(durations)
        remain = (total - i) * avg
        m, s = divmod(int(remain), 60)
        sys.stdout.write(f"\r[{i}/{total}] {status} | ETA {m:02d}:{s:02d}")
        sys.stdout.flush()
    print("\nDownload complete.")

def make_video(image_dir, output_path, fps=30):
    files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
    if not files:
        print("No images found.")
        return

    list_file = os.path.join(image_dir, "imagelist.txt")
    with open(list_file, "w") as f:
        for fn in files:
            f.write(f"file '{os.path.join(image_dir, fn)}'\n")
            f.write(f"duration {1/fps:.2f}\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-vsync", "vfr",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    subprocess.run(cmd)
    print("Video saved →", output_path)

def main():
    base_dir = input("Enter base directory path: ").strip()
    os.makedirs(base_dir, exist_ok=True)

    start_code = "20252991716"
    end_code   = "20253021046"
    prefix     = "_GOES19-ABI-CONUS-GEOCOLOR-1250x750.jpg"
    base_url   = "https://cdn.star.nesdis.noaa.gov/GOES19/ABI/CONUS/GEOCOLOR/"

    csv_dir   = os.path.join(base_dir, "csv")
    img_dir   = os.path.join(base_dir, "images")
    vid_dir   = os.path.join(base_dir, "video")

    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(vid_dir, exist_ok=True)

    csv_path  = os.path.join(csv_dir, "goes19_filenames.csv")
    video_out = os.path.join(vid_dir, "goes19_timelapse.mp4")

    filenames = generate_filenames(start_code, end_code, prefix, csv_path)
    download_images(filenames, img_dir, base_url)
    make_video(img_dir, video_out, fps=30)

if __name__ == "__main__":
    main()
