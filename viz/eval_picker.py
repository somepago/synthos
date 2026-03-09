#!/usr/bin/env python3
"""
Streamlit app to browse images from local datasets and pick them for eval set.

Usage: streamlit run viz/eval_picker.py --server.port 8505
"""

import io
import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

DATA_ROOT = Path("/home/gnan/projects/data")
EVAL_DIR = Path("/home/gnan/projects/synthos/eval_unified")
PICKS_FILE = EVAL_DIR / "picks.json"

THUMB_SIZE = 256  # max dimension for thumbnails

DATASETS = {
    "relaion-art-lowres": {
        "parquet": DATA_ROOT / "datasets" / "relaion-art-lowres" / "downloaded.parquet",
        "images": DATA_ROOT / "datasets" / "relaion-art-lowres" / "images",
        "status_col": "download_status",
        "caption_cols": ["text"],
    },
    "relaion-pop": {
        "parquet": DATA_ROOT / "datasets" / "laion__relaion-pop" / "downloaded.parquet",
        "images": DATA_ROOT / "datasets" / "laion__relaion-pop" / "images",
        "status_col": "download_status",
        "caption_cols": ["cogvlm_caption", "llava_caption", "alt_txt"],
    },
    "midjourney-top": {
        "parquet": DATA_ROOT / "scrapes" / "midjourney-top" / "downloaded.parquet",
        "images": DATA_ROOT / "scrapes" / "midjourney-top" / "images",
        "status_col": "download_status",
        "caption_cols": ["prompt"],
    },
    "wikiart": {
        "parquet": DATA_ROOT / "datasets" / "huggan__wikiart" / "metadata.parquet",
        "images": DATA_ROOT / "datasets" / "huggan__wikiart" / "images",
        "status_col": None,
        "caption_cols": ["artist", "genre", "style"],
        "file_name_col": "_index",
    },
}


def load_thumbnail(img_path):
    """Load image as small JPEG thumbnail for fast display."""
    try:
        img = Image.open(img_path)
        img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        return buf.getvalue()
    except Exception:
        return None


@st.cache_data
def load_dataset(name):
    cfg = DATASETS[name]
    df = pd.read_parquet(cfg["parquet"])
    if cfg.get("status_col"):
        df = df[df[cfg["status_col"]] == "ok"].reset_index(drop=True)
    fn_col = cfg.get("file_name_col", "file_name")
    if fn_col != "file_name":
        df["file_name"] = df[fn_col].astype(str) + ".jpg"
    df["_dataset"] = name
    return df


def load_picks():
    if PICKS_FILE.exists():
        return json.loads(PICKS_FILE.read_text())
    return []


def save_picks(picks):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PICKS_FILE.write_text(json.dumps(picks, indent=2))


def get_image_path(dataset_name, file_name):
    return DATASETS[dataset_name]["images"] / file_name


def main():
    st.set_page_config(page_title="Eval Image Picker", layout="wide")
    st.title("Eval Set Builder")

    picks = load_picks()

    # Sidebar
    st.sidebar.header(f"Picked: {len(picks)} images")
    if st.sidebar.button("Export eval set"):
        n_new = export_eval_set(picks)
        st.sidebar.success(f"Exported {n_new} new images to {EVAL_DIR}/images/")

    # Main area
    tab_browse, tab_picks = st.tabs(["Browse", f"Picked ({len(picks)})"])

    with tab_browse:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            dataset_name = st.selectbox("Dataset", list(DATASETS.keys()))
        with col2:
            n_show = st.slider("Images per batch", 20, 200, 100, step=20)
        with col3:
            n_cols = st.slider("Columns", 4, 10, 6)

        df = load_dataset(dataset_name)
        cfg = DATASETS[dataset_name]
        st.caption(f"{len(df)} images available")

        search = st.text_input("Search captions (optional)")
        if search:
            mask = pd.Series(False, index=df.index)
            for col in cfg["caption_cols"]:
                if col in df.columns:
                    mask |= df[col].fillna("").str.contains(search, case=False)
            df_filtered = df[mask]
            st.write(f"Found {len(df_filtered)} matches")
        else:
            df_filtered = df

        if len(df_filtered) == 0:
            st.warning("No images found")
            return

        # Pagination
        page_count = max(1, (len(df_filtered) + n_show - 1) // n_show)
        col_shuf, col_page = st.columns([1, 3])
        with col_shuf:
            if st.button("Shuffle"):
                import random
                st.session_state.batch_seed = random.randint(0, 999999)
                st.session_state.page = 0
            if "batch_seed" not in st.session_state:
                import random
                st.session_state.batch_seed = random.randint(0, 999999)
        with col_page:
            page = st.number_input("Page", 0, page_count - 1,
                                   st.session_state.get("page", 0), key="page_input")
            st.session_state.page = page

        # Sample or paginate
        if search:
            start = page * n_show
            batch = df_filtered.iloc[start:start + n_show]
        else:
            shuffled = df_filtered.sample(frac=1, random_state=st.session_state.batch_seed)
            start = page * n_show
            batch = shuffled.iloc[start:start + n_show]

        picked_fnames = {p["file_name"] for p in picks}
        cols = st.columns(n_cols)
        for idx, (_, row) in enumerate(batch.iterrows()):
            col = cols[idx % n_cols]
            img_path = get_image_path(dataset_name, row["file_name"])
            if not img_path.exists():
                continue

            with col:
                thumb = load_thumbnail(img_path)
                if thumb is None:
                    continue
                st.image(thumb, use_container_width=True)

                for cap_col in cfg["caption_cols"]:
                    if cap_col in row and pd.notna(row[cap_col]):
                        st.caption(str(row[cap_col])[:80])
                        break

                if row["file_name"] in picked_fnames:
                    st.success("Picked!")
                else:
                    if st.button("Pick", key=f"pick_{row['file_name']}"):
                        picks.append({
                            "file_name": row["file_name"],
                            "dataset": dataset_name,
                            "width": int(row.get("width", 0)),
                            "height": int(row.get("height", 0)),
                        })
                        save_picks(picks)
                        st.rerun()

    with tab_picks:
        if not picks:
            st.info("No images picked yet")
        else:
            cols = st.columns(6)
            for idx, p in enumerate(picks):
                col = cols[idx % 6]
                img_path = get_image_path(p["dataset"], p["file_name"])
                with col:
                    if img_path.exists():
                        thumb = load_thumbnail(img_path)
                        if thumb:
                            st.image(thumb, use_container_width=True)
                    st.caption(f"{p['dataset']} | {p.get('width', '?')}x{p.get('height', '?')}")
                    if st.button("Remove", key=f"rm_{idx}"):
                        picks.pop(idx)
                        save_picks(picks)
                        st.rerun()


def export_eval_set(picks):
    images_dir = EVAL_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # Continue numbering from existing images
    existing = sorted(images_dir.glob("*.*"))
    next_id = max((int(f.stem) for f in existing if f.stem.isdigit()), default=-1) + 1

    exported = []
    picked_fnames = set()
    # Check what's already in the folder by orig_name from eval.csv
    csv_path = EVAL_DIR / "eval.csv"
    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                picked_fnames.add(row.get("orig_name", ""))

    for p in picks:
        if p["file_name"] in picked_fnames:
            continue  # already exported
        src = get_image_path(p["dataset"], p["file_name"])
        if not src.exists():
            continue
        ext = src.suffix
        dst = images_dir / f"{next_id:03d}{ext}"
        shutil.copy2(src, dst)
        exported.append({**p, "eval_name": dst.name, "id": next_id})
        next_id += 1

    # Append to eval.csv
    if exported and csv_path.exists():
        import csv
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            for e in exported:
                writer.writerow([
                    e["id"], f"images/{e['eval_name']}", e["file_name"],
                    e["dataset"], "real" if e["dataset"] != "midjourney-top" else "generated",
                    e.get("width", ""), e.get("height", ""), "", "",
                ])

    return len(exported)


if __name__ == "__main__":
    main()
