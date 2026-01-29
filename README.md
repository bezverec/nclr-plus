# nclr-plus: NDK Color Plus

**nclr-plus** is a Rust CLI for “document-friendly” enhancement (and optional 2× upscale) of **RGB TIFF scans**, while preserving **DPI** and **embedded ICC profiles** and performing color management with **LittleCMS2**.

It’s aimed at common digitization outputs (books / newspapers) where you want:
- a bit more perceived detail / microcontrast,
- stable colors (chroma denoise + ΔE limiter),
- optional shading correction (page background / uneven lighting),
- and *not* break metadata that libraries/validators care about (ICC + resolution tags).

## Note
This tool does **not** replace properly calibrated book scanners or controlled digitization workflows.  
It should not be used on archival (master) scans except as a last resort.

This tool adds more features to a more conservative **NCLR** tool (color management only), see https://github.com/bezverec/nclr

## AI generated code disclosure

The code is AI generated using ChatGPT model 5.2

---

## Features

- **Batch processing**: input can be a single TIFF or a directory tree (WalkDir + Rayon).
- **Presets**: `book`, `newsprint` (different sharpening/denoise/shading defaults).
- **Upscale**: `--scale 1|2` (Lanczos3).
- **Optional shading correction**: `--shading` (useful for bound volumes).
- **ICC-aware pipeline** (lcms2):
  - auto preserve embedded ICC (or fallback to sRGB),
  - force sRGB output,
  - or disable embedding.
- **Output bit depth**: `--out-depth b16|b8`, optional Floyd–Steinberg dithering for 8-bit.
- **White point selection for Lab + ΔE limiter**: `--white-point d65|d50`  
  (D50 uses Bradford adaptation in Lab conversions).
- **ΔE limiter**: `--delta-e none|cie76|ciede2000` with `--delta-e-max` (default 2.5)  
  to reduce “too strong” local changes.

---

## Input / output formats

### Currently supported input
- **TIFF (.tif / .tiff)** only.

### Output
- **TIFF** (RGB, 16-bit or 8-bit depending on `--out-depth`)
- Writes ICC as TIFF tag **34675 (ICCProfile)** and preserves ResolutionUnit/XResolution/YResolution.

---

## Build from source

### Prerequisites
1. Install [Git](https://git-scm.com/)
2. Install [**Rust** (stable)](https://www.rust-lang.org/tools/install) and Cargo

### Compilation

```bash
git clone https://github.com/bezverec/nclr-plus
```
```
cd nclr-plus
```
Linux:
```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```
Windows:
```powershell
$env:RUSTFLAGS="-C target-cpu=native"; cargo build --release
```

### Run the binary

- Linux/macOS:
  ```bash
  ./target/release/nclr-plus --help
  ```
- Windows (PowerShell):
  ```powershell
  .\target\release\nclr-plus.exe --help
  ```

### `cargo install` (local path)

```bash
cargo install --path .
```

---

## Usage

### Single file

```bash
nclr-plus INPUT.tif OUT_DIR --preset book --shading
```

### Directory (recursive)

```bash
nclr-plus INPUT_DIR OUT_DIR --preset newsprint --parallel
```

### 2× upscale but keep *physical size* (DPI scaled)

```bash
nclr-plus INPUT_DIR OUT_DIR --scale 2 --dpi-scale true
```

### Force sRGB output and embed sRGB ICC

```bash
nclr-plus INPUT_DIR OUT_DIR --icc-mode srgb
```

### Output 8-bit TIFF + dithering

```bash
nclr-plus INPUT_DIR OUT_DIR --out-depth b8 --dither
```

### Use D50 Lab + stricter ΔE limiter

```bash
nclr-plus INPUT_DIR OUT_DIR --white-point d50 --delta-e ciede2000 --delta-e-max 2.0
```

---

## CLI options (overview)

- `--preset {book|newsprint}`
- `--scale {1|2}`
- `--shading`
- `--dpi-scale {true|false}` (default `true`)
- `--icc-mode {auto|srgb|none}`
- `--out-icc PATH` (explicit output ICC profile)
- `--intent {perceptual|relative|absolute|saturation}`
- `--bpc {true|false}` (black point compensation, default `true`)
- `--no-icc` (skip transforms; treat input as sRGB)
- `--out-depth {b16|b8}`
- `--dither` (for 8-bit output)
- `--white-point {d65|d50}`
- `--delta-e {none|cie76|ciede2000}`
- `--delta-e-max FLOAT` (default `2.5`)
- `--force` (overwrite outputs)
- `--parallel {true|false}`
  
---

## Practical tips (digitization workflow)

- **Books / uneven lighting**: try `--preset book --shading`.
- **Newspapers**: start with defaults (`newsprint`) and consider lowering `--delta-e-max` if you see local “snap”.
- **If color is critical**: keep `--out-depth b16` and don’t dither.
- **If you need smaller files**: use `--out-depth b8 --dither` (still uncompressed TIFF in the current implementation).

---

## Performance

- Uses Rayon for parallelism (directories) and a release profile tuned for speed/size
  (`lto = true`, `codegen-units = 1`, `panic = "abort"`).
- Processing is CPU-heavy (Gaussian blurs + Sobel + Lab conversions).

---

## Limitations / TODO

- TIFF-only input for now (JPG/JP2 planned).
- Currently assumes **RGB** input (no grayscale/CMYK/alpha).
- No compression controls for TIFF output yet (could be added: Deflate/LZW/ZSTD depending on encoder support).

---

## License

**GPL 3.0**. See `LICENSE`.
