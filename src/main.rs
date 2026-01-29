use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, ValueEnum};
use image::imageops::FilterType;
use image::ImageBuffer;
use imageproc::filter::gaussian_blur_f32;
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use lcms2::{Flags, Intent, PixelFormat, Profile, Transform};
use palette::white_point::{D50, D65};
use palette::{IntoColor, Lab, LinSrgb, Srgb, Xyz};
use palette::chromatic_adaptation::{AdaptInto, Method};

use rayon::prelude::*;
use rgb::{RGB16, RGB8};
use std::{
    borrow::Cow,
    fs,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};
use tiff::{
    decoder::{DecodingResult, Decoder},
    encoder::{colortype, Rational as TRational, TiffEncoder, TiffValue},
    tags::{ResolutionUnit, Tag, Type as TiffType},
};
use walkdir::WalkDir;

type LabD65 = Lab<D65, f32>;
type LabD50 = Lab<D50, f32>;

const TAG_ICC_PROFILE_U16: u16 = 34675;
const TAG_ICC_PROFILE: Tag = Tag::Unknown(TAG_ICC_PROFILE_U16);

#[derive(Parser, Debug)]
#[command(
    name="docfsr",
    about="Document-friendly enhance/upscale for TIFF scans (ICC + DPI preserved, lcms2)."
)]
struct Cli {
    /// Input TIFF file or directory
    #[arg(value_name="INPUT")]
    input: PathBuf,

    /// Output directory (created if missing)
    #[arg(value_name="OUTPUT_DIR")]
    output_dir: PathBuf,

    /// Preset
    #[arg(long, value_enum, default_value="newsprint")]
    preset: Preset,

    /// Scale factor (1 or 2)
    #[arg(long, default_value_t=1)]
    scale: u32,

    /// Shading correction (books often benefit)
    #[arg(long)]
    shading: bool,

    /// If scaling, multiply X/Y resolution numerators by scale to keep physical size same
    #[arg(long, default_value_t=true)]
    dpi_scale: bool,

    /// ICC handling:
    /// - auto: if embedded ICC exists -> preserve it as output; else output sRGB
    /// - srgb: always output sRGB
    /// - none: do not embed ICC in output
    #[arg(long, value_enum, default_value="auto")]
    icc_mode: IccMode,

    /// Explicit output ICC profile file (overrides icc_mode auto/srgb)
    #[arg(long)]
    out_icc: Option<PathBuf>,

    /// Rendering intent (ICC transform)
    #[arg(long, value_enum, default_value="perceptual")]
    intent: RenderIntent,

    /// Black Point Compensation
    #[arg(long, default_value_t=true)]
    bpc: bool,

    /// If set, skip ICC transforms (treat input as sRGB), but still preserve/emit ICC by icc_mode/out_icc
    #[arg(long, default_value_t=false)]
    no_icc: bool,

    /// Output depth (TIFF)
    #[arg(long, value_enum, default_value="b16")]
    out_depth: OutDepth,

    /// Dither when writing 8-bit output
    #[arg(long, default_value_t=false)]
    dither: bool,

    /// White point used for Lab + ΔE limiter computations
    #[arg(long, value_enum, default_value="d65")]
    white_point: WhitePoint,

    /// ΔE limiter mode to prevent shifts (computed in Lab)
    #[arg(long, value_enum, default_value="ciede2000")]
    delta_e: DeltaEMode,

    /// Max allowed ΔE; effect is reduced above this
    #[arg(long, default_value_t=2.5)]
    delta_e_max: f32,

    /// Overwrite outputs
    #[arg(long)]
    force: bool,

    /// Parallel processing for directories
    #[arg(long, default_value_t=true)]
    parallel: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preset {
    Book,
    Newsprint,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum IccMode {
    Auto,
    Srgb,
    None,
}

#[derive(Debug, Copy, Clone, ValueEnum)]
enum RenderIntent {
    Perceptual,
    Relative,
    Absolute,
    Saturation,
}
impl From<RenderIntent> for Intent {
    fn from(v: RenderIntent) -> Self {
        match v {
            RenderIntent::Perceptual => Intent::Perceptual,
            RenderIntent::Relative => Intent::RelativeColorimetric,
            RenderIntent::Absolute => Intent::AbsoluteColorimetric,
            RenderIntent::Saturation => Intent::Saturation,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum OutDepth {
    B8,
    B16,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum WhitePoint {
    D50,
    D65,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeltaEMode {
    None,
    Cie76,
    Ciede2000,
}

#[derive(Clone, Debug)]
struct Params {
    shading_sigma: f32,
    shading_strength: f32,
    chroma_sigma: f32,
    base_sigma: f32,
    detail_sigma: f32,
    sharpen_strength: f32,
    sharpen_blur_sigma: f32,
    limiter_16bit: f32, // in "16-bit units"
    edge_thresh: f32,
    edge_softness: f32,
}

impl Params {
    fn for_preset(p: Preset) -> Self {
        match p {
            Preset::Book => Self {
                shading_sigma: 28.0,
                shading_strength: 0.55,
                chroma_sigma: 1.0,
                base_sigma: 1.4,
                detail_sigma: 0.6,
                sharpen_strength: 0.25,
                sharpen_blur_sigma: 1.0,
                limiter_16bit: 768.0, // ~3/255 of 65535
                edge_thresh: 0.10,
                edge_softness: 0.06,
            },
            Preset::Newsprint => Self {
                shading_sigma: 22.0,
                shading_strength: 0.35,
                chroma_sigma: 0.8,
                base_sigma: 1.1,
                detail_sigma: 0.35,
                sharpen_strength: 0.18,
                sharpen_blur_sigma: 0.9,
                limiter_16bit: 512.0, // ~2/255 of 65535
                edge_thresh: 0.12,
                edge_softness: 0.06,
            },
        }
    }
}

// ---------------- TIFF meta (minimal reader: ICC + DPI) ----------------

#[derive(Clone, Debug, Default)]
struct Rational {
    n: u32,
    d: u32,
}

#[derive(Clone, Debug, Default)]
struct TiffMeta {
    icc: Option<Vec<u8>>,
    x_res: Option<Rational>,
    y_res: Option<Rational>,
    unit: Option<ResolutionUnit>,
}

fn file_ext_lower(p: &Path) -> String {
    p.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn is_tiff(p: &Path) -> bool {
    matches!(file_ext_lower(p).as_str(), "tif" | "tiff")
}

fn read_exact_at(f: &mut File, off: u64, buf: &mut [u8]) -> Result<()> {
    f.seek(SeekFrom::Start(off))
        .with_context(|| format!("Seek @ {off}"))?;
    f.read_exact(buf)
        .with_context(|| format!("Read {} bytes @ {off}", buf.len()))?;
    Ok(())
}

fn read_u16_endian(b: [u8; 2], le: bool) -> u16 {
    if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) }
}
fn read_u32_endian(b: [u8; 4], le: bool) -> u32 {
    if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) }
}
fn read_u64_endian(b: [u8; 8], le: bool) -> u64 {
    if le { u64::from_le_bytes(b) } else { u64::from_be_bytes(b) }
}

/// Minimal TIFF/BigTIFF reader for:
/// - ICCProfile (34675)
/// - XResolution (282), YResolution (283), ResolutionUnit (296)
fn read_tiff_meta(path: &Path) -> Result<TiffMeta> {
    let mut f = File::open(path).with_context(|| format!("Open TIFF: {}", path.display()))?;

    let mut head = [0u8; 16];
    read_exact_at(&mut f, 0, &mut head)?;

    let le = match &head[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => bail!("Not a TIFF (bad endian marker)"),
    };

    let magic = read_u16_endian([head[2], head[3]], le);

    fn type_size(t: u16) -> Option<u64> {
        match t {
            1 => Some(1),  // BYTE
            3 => Some(2),  // SHORT
            4 => Some(4),  // LONG
            5 => Some(8),  // RATIONAL (2x u32)
            7 => Some(1),  // UNDEFINED
            16 => Some(8), // LONG8 (BigTIFF)
            _ => None,
        }
    }

    let mut meta = TiffMeta::default();

    let icc_tag: u16 = TAG_ICC_PROFILE_U16;
    let xres_tag: u16 = 282;
    let yres_tag: u16 = 283;
    let unit_tag: u16 = 296;

    if magic == 42 {
        let ifd0_off = read_u32_endian([head[4], head[5], head[6], head[7]], le) as u64;

        let mut nbuf = [0u8; 2];
        read_exact_at(&mut f, ifd0_off, &mut nbuf)?;
        let n = read_u16_endian(nbuf, le) as u64;

        let mut ent_off = ifd0_off + 2;
        for _ in 0..n {
            let mut ent = [0u8; 12];
            read_exact_at(&mut f, ent_off, &mut ent)?;
            ent_off += 12;

            let tag = read_u16_endian([ent[0], ent[1]], le);
            let ty = read_u16_endian([ent[2], ent[3]], le);
            let count = read_u32_endian([ent[4], ent[5], ent[6], ent[7]], le) as u64;
            let value_or_off = read_u32_endian([ent[8], ent[9], ent[10], ent[11]], le) as u64;

            let tsz = match type_size(ty) {
                Some(s) => s,
                None => continue,
            };
            let bytes_len = count.saturating_mul(tsz);

            let get_bytes = |f: &mut File| -> Result<Vec<u8>> {
                if bytes_len == 0 {
                    return Ok(Vec::new());
                }
                if bytes_len <= 4 {
                    Ok(ent[8..8 + (bytes_len as usize)].to_vec())
                } else {
                    let mut v = vec![0u8; bytes_len as usize];
                    read_exact_at(f, value_or_off, &mut v)?;
                    Ok(v)
                }
            };

            match tag {
                t if t == icc_tag => {
                    let b = get_bytes(&mut f)?;
                    if !b.is_empty() {
                        meta.icc = Some(b);
                    }
                }
                t if t == xres_tag => {
                    let b = get_bytes(&mut f)?;
                    if b.len() >= 8 {
                        let n = read_u32_endian([b[0], b[1], b[2], b[3]], le);
                        let d = read_u32_endian([b[4], b[5], b[6], b[7]], le);
                        if d != 0 {
                            meta.x_res = Some(Rational { n, d });
                        }
                    }
                }
                t if t == yres_tag => {
                    let b = get_bytes(&mut f)?;
                    if b.len() >= 8 {
                        let n = read_u32_endian([b[0], b[1], b[2], b[3]], le);
                        let d = read_u32_endian([b[4], b[5], b[6], b[7]], le);
                        if d != 0 {
                            meta.y_res = Some(Rational { n, d });
                        }
                    }
                }
                t if t == unit_tag => {
                    let b = get_bytes(&mut f)?;
                    if b.len() >= 2 {
                        let u = read_u16_endian([b[0], b[1]], le);
                        meta.unit = Some(match u {
                            2 => ResolutionUnit::Inch,
                            3 => ResolutionUnit::Centimeter,
                            _ => ResolutionUnit::None,
                        });
                    }
                }
                _ => {}
            }
        }
    } else if magic == 43 {
        let off_size = read_u16_endian([head[4], head[5]], le);
        if off_size != 8 {
            bail!("Unsupported BigTIFF offset size: {}", off_size);
        }
        let ifd0_off = read_u64_endian(
            [
                head[8], head[9], head[10], head[11], head[12], head[13], head[14], head[15],
            ],
            le,
        );

        let mut nbuf = [0u8; 8];
        read_exact_at(&mut f, ifd0_off, &mut nbuf)?;
        let n = read_u64_endian(nbuf, le);

        let mut ent_off = ifd0_off + 8;
        for _ in 0..n {
            let mut ent = [0u8; 20];
            read_exact_at(&mut f, ent_off, &mut ent)?;
            ent_off += 20;

            let tag = read_u16_endian([ent[0], ent[1]], le);
            let ty = read_u16_endian([ent[2], ent[3]], le);
            let count = read_u64_endian(
                [ent[4], ent[5], ent[6], ent[7], ent[8], ent[9], ent[10], ent[11]],
                le,
            );
            let value_or_off = read_u64_endian(
                [ent[12], ent[13], ent[14], ent[15], ent[16], ent[17], ent[18], ent[19]],
                le,
            );

            let tsz = match type_size(ty) {
                Some(s) => s,
                None => continue,
            };
            let bytes_len = count.saturating_mul(tsz);

            let get_bytes = |f: &mut File| -> Result<Vec<u8>> {
                if bytes_len == 0 {
                    return Ok(Vec::new());
                }
                if bytes_len <= 8 {
                    Ok(ent[12..12 + (bytes_len as usize)].to_vec())
                } else {
                    let mut v = vec![0u8; bytes_len as usize];
                    read_exact_at(f, value_or_off, &mut v)?;
                    Ok(v)
                }
            };

            match tag {
                t if t == icc_tag => {
                    let b = get_bytes(&mut f)?;
                    if !b.is_empty() {
                        meta.icc = Some(b);
                    }
                }
                t if t == xres_tag => {
                    let b = get_bytes(&mut f)?;
                    if b.len() >= 8 {
                        let n = read_u32_endian([b[0], b[1], b[2], b[3]], le);
                        let d = read_u32_endian([b[4], b[5], b[6], b[7]], le);
                        if d != 0 {
                            meta.x_res = Some(Rational { n, d });
                        }
                    }
                }
                t if t == yres_tag => {
                    let b = get_bytes(&mut f)?;
                    if b.len() >= 8 {
                        let n = read_u32_endian([b[0], b[1], b[2], b[3]], le);
                        let d = read_u32_endian([b[4], b[5], b[6], b[7]], le);
                        if d != 0 {
                            meta.y_res = Some(Rational { n, d });
                        }
                    }
                }
                t if t == unit_tag => {
                    let b = get_bytes(&mut f)?;
                    if b.len() >= 2 {
                        let u = read_u16_endian([b[0], b[1]], le);
                        meta.unit = Some(match u {
                            2 => ResolutionUnit::Inch,
                            3 => ResolutionUnit::Centimeter,
                            _ => ResolutionUnit::None,
                        });
                    }
                }
                _ => {}
            }
        }
    } else {
        bail!("Unknown TIFF magic: {}", magic);
    }

    Ok(meta)
}

// ---------------- ICC writing helper: UNDEFINED bytes ----------------

struct UndefinedBytes<'a>(&'a [u8]);

impl<'a> TiffValue for UndefinedBytes<'a> {
    const BYTE_LEN: u8 = 1;
    const FIELD_TYPE: TiffType = TiffType::UNDEFINED;

    fn count(&self) -> usize {
        self.0.len()
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(self.0)
    }
}

fn normalize_resolution(meta: &TiffMeta) -> (ResolutionUnit, TRational, TRational) {
    let mut unit = meta.unit.unwrap_or(ResolutionUnit::Inch);
    if matches!(unit, ResolutionUnit::None) {
        unit = ResolutionUnit::Inch;
    }

    // Build from primitives so we can mirror without move issues.
    let mut xr_n = 600u32;
    let mut xr_d = 1u32;
    let mut yr_n = 600u32;
    let mut yr_d = 1u32;

    if let Some(x) = &meta.x_res {
        if x.d != 0 {
            xr_n = x.n;
            xr_d = x.d;
        }
    }
    if let Some(y) = &meta.y_res {
        if y.d != 0 {
            yr_n = y.n;
            yr_d = y.d;
        }
    } else if meta.x_res.is_some() {
        yr_n = xr_n;
        yr_d = xr_d;
    }

    if meta.x_res.is_none() && meta.y_res.is_some() {
        xr_n = yr_n;
        xr_d = yr_d;
    }

    if xr_d == 0 {
        xr_n = 600;
        xr_d = 1;
    }
    if yr_d == 0 {
        yr_n = 600;
        yr_d = 1;
    }

    (
        unit,
        TRational { n: xr_n, d: xr_d },
        TRational { n: yr_n, d: yr_d },
    )
}

fn read_tiff_rgb16(path: &Path) -> Result<(u32, u32, Vec<RGB16>)> {
    let f = File::open(path)?;
    let mut dec = Decoder::new(BufReader::new(f))?;
    let (w, h) = dec.dimensions()?;

    let pix = match dec.read_image()? {
        DecodingResult::U16(v) => {
            if v.len() != (w as usize) * (h as usize) * 3 {
                bail!(
                    "Unexpected TIFF samples count (expected RGB16 interleaved): {}",
                    v.len()
                );
            }
            v.chunks_exact(3)
                .map(|c| RGB16::new(c[0], c[1], c[2]))
                .collect::<Vec<_>>()
        }
        DecodingResult::U8(v8) => {
            if v8.len() != (w as usize) * (h as usize) * 3 {
                bail!(
                    "Unexpected TIFF samples count (expected RGB8 interleaved): {}",
                    v8.len()
                );
            }
            v8.chunks_exact(3)
                .map(|c| RGB16::new((c[0] as u16) << 8, (c[1] as u16) << 8, (c[2] as u16) << 8))
                .collect::<Vec<_>>()
        }
        other => return Err(anyhow!("Unsupported TIFF decoding result: {:?}", other)),
    };

    Ok((w, h, pix))
}

fn write_tiff_rgb16(
    out_path: &Path,
    w: u32,
    h: u32,
    pix: &[RGB16],
    meta: &TiffMeta,
    icc_bytes: Option<&[u8]>,
) -> Result<()> {
    let f = File::create(out_path).with_context(|| format!("Create output: {}", out_path.display()))?;
    let mut tiff = TiffEncoder::new(BufWriter::new(f))?;
    let mut img = tiff.new_image::<colortype::RGB16>(w, h)?;

    let (unit, xr, yr) = normalize_resolution(meta);
    img.resolution_unit(unit);
    img.x_resolution(xr);
    img.y_resolution(yr);

    if let Some(icc) = icc_bytes {
        img.encoder()
            .write_tag(TAG_ICC_PROFILE, UndefinedBytes(icc))
            .context("Write ICCProfile tag (34675) as UNDEFINED")?;
    }

    img.rows_per_strip(64)?;

    let mut row = 0u32;
    while img.next_strip_sample_count() > 0 {
        let rows = (h - row).min(64);
        let start = (row as usize) * (w as usize);
        let end = ((row + rows) as usize) * (w as usize);
        let slice = &pix[start..end];

        let mut raw: Vec<u16> = Vec::with_capacity(slice.len() * 3);
        for p in slice {
            raw.push(p.r);
            raw.push(p.g);
            raw.push(p.b);
        }
        img.write_strip(&raw)?;
        row += rows;
    }

    img.finish()?;
    Ok(())
}

fn write_tiff_rgb8(
    out_path: &Path,
    w: u32,
    h: u32,
    pix: &[RGB8],
    meta: &TiffMeta,
    icc_bytes: Option<&[u8]>,
) -> Result<()> {
    let f = File::create(out_path).with_context(|| format!("Create output: {}", out_path.display()))?;
    let mut tiff = TiffEncoder::new(BufWriter::new(f))?;
    let mut img = tiff.new_image::<colortype::RGB8>(w, h)?;

    let (unit, xr, yr) = normalize_resolution(meta);
    img.resolution_unit(unit);
    img.x_resolution(xr);
    img.y_resolution(yr);

    if let Some(icc) = icc_bytes {
        img.encoder()
            .write_tag(TAG_ICC_PROFILE, UndefinedBytes(icc))
            .context("Write ICCProfile tag (34675) as UNDEFINED")?;
    }

    img.rows_per_strip(128)?;

    let mut row = 0u32;
    while img.next_strip_sample_count() > 0 {
        let rows = (h - row).min(128);
        let start = (row as usize) * (w as usize);
        let end = ((row + rows) as usize) * (w as usize);
        let slice = &pix[start..end];

        let mut raw: Vec<u8> = Vec::with_capacity(slice.len() * 3);
        for p in slice {
            raw.push(p.r);
            raw.push(p.g);
            raw.push(p.b);
        }
        img.write_strip(&raw)?;
        row += rows;
    }

    img.finish()?;
    Ok(())
}

// ---------------- ICC policy + transforms ----------------

fn pick_input_profile(meta: &TiffMeta) -> Result<Profile> {
    if let Some(bytes) = meta.icc.as_deref() {
        Ok(Profile::new_icc(bytes)?)
    } else {
        Ok(Profile::new_srgb())
    }
}

fn pick_output_profile(cli: &Cli, meta: &TiffMeta) -> Result<Option<Profile>> {
    if let Some(p) = cli.out_icc.as_deref() {
        return Ok(Some(Profile::new_file(p)?));
    }

    match cli.icc_mode {
        IccMode::None => Ok(None),
        IccMode::Srgb => Ok(Some(Profile::new_srgb())),
        IccMode::Auto => {
            if let Some(bytes) = meta.icc.as_deref() {
                Ok(Some(Profile::new_icc(bytes)?))
            } else {
                Ok(Some(Profile::new_srgb()))
            }
        }
    }
}

fn build_transform_rgb16(
    in_prof: &Profile,
    out_prof: &Profile,
    intent: Intent,
    bpc: bool,
) -> Result<Transform<RGB16, RGB16>> {
    let mut flags = Flags::default();
    if bpc {
        // Flags doesn't implement |= ; use normal assignment.
        flags = flags | Flags::BLACKPOINT_COMPENSATION;
    }
    Ok(Transform::new_flags(
        in_prof,
        PixelFormat::RGB_16,
        out_prof,
        PixelFormat::RGB_16,
        intent,
        flags,
    )?)
}

// ---------------- ΔE limiter helpers ----------------

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn delta_e(mode: DeltaEMode, l1: (f32, f32, f32), l2: (f32, f32, f32)) -> f32 {
    match mode {
        DeltaEMode::None => 0.0,
        DeltaEMode::Cie76 => {
            let dl = l1.0 - l2.0;
            let da = l1.1 - l2.1;
            let db = l1.2 - l2.2;
            (dl * dl + da * da + db * db).sqrt()
        }
        DeltaEMode::Ciede2000 => ciede2000(l1, l2),
    }
}

// CIEDE2000 implementation (standard formula)
// Inputs: Lab L* in [0..100], a*, b* roughly [-128..128]
fn ciede2000(lab1: (f32, f32, f32), lab2: (f32, f32, f32)) -> f32 {

    let (l1, a1, b1) = lab1;
    let (l2, a2, b2) = lab2;

    let k_l = 1.0f32;
    let k_c = 1.0f32;
    let k_h = 1.0f32;

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = (c1 + c2) * 0.5;

    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + 25.0f32.powi(7))).sqrt());

    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;

    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();

    let h1p = hue_angle_deg(a1p, b1);
    let h2p = hue_angle_deg(a2p, b2);

    let dlp = l2 - l1;
    let dcp = c2p - c1p;

    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else {
        let mut dh = h2p - h1p;
        if dh > 180.0 {
            dh -= 360.0;
        } else if dh < -180.0 {
            dh += 360.0;
        }
        dh
    };

    let dhp_rad = (dhp.to_radians()) * 0.5;
    let dh_term = 2.0 * (c1p * c2p).sqrt() * dhp_rad.sin();

    let l_bar_p = (l1 + l2) * 0.5;
    let c_bar_p = (c1p + c2p) * 0.5;

    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else {
        let mut hsum = h1p + h2p;
        let hdiff = (h1p - h2p).abs();
        if hdiff > 180.0 {
            if hsum < 360.0 {
                hsum += 360.0;
            }
            hsum * 0.5
        } else {
            hsum * 0.5
        }
    };

    let t = 1.0
        - 0.17 * ((h_bar_p - 30.0).to_radians()).cos()
        + 0.24 * ((2.0 * h_bar_p).to_radians()).cos()
        + 0.32 * ((3.0 * h_bar_p + 6.0).to_radians()).cos()
        - 0.20 * ((4.0 * h_bar_p - 63.0).to_radians()).cos();

    let delta_theta = 30.0 * (-( (h_bar_p - 275.0) / 25.0 ).powi(2)).exp();
    let r_c = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + 25.0f32.powi(7))).sqrt();

    let s_l = 1.0 + (0.015 * (l_bar_p - 50.0).powi(2)) / (20.0 + (l_bar_p - 50.0).powi(2)).sqrt();
    let s_c = 1.0 + 0.045 * c_bar_p;
    let s_h = 1.0 + 0.015 * c_bar_p * t;

    let r_t = -r_c * (2.0 * delta_theta.to_radians()).sin();

    let dl = dlp / (k_l * s_l);
    let dc = dcp / (k_c * s_c);
    let dh = dh_term / (k_h * s_h);

    (dl * dl + dc * dc + dh * dh + r_t * dc * dh).sqrt()
}

fn hue_angle_deg(a: f32, b: f32) -> f32 {
    use std::f32::consts::PI;
    if a == 0.0 && b == 0.0 {
        return 0.0;
    }
    let mut h = b.atan2(a) * 180.0 / PI;
    if h < 0.0 {
        h += 360.0;
    }
    h
}

// ---------------- Quantization (16->8) with optional dithering ----------------

fn quantize_rgb16_to_rgb8(pix: &[RGB16], w: u32, h: u32, dither: bool) -> Vec<RGB8> {
    let w = w as usize;
    let h = h as usize;
    let n = w * h;
    let mut out = vec![RGB8::new(0, 0, 0); n];

    if !dither {
        for i in 0..n {
            let p = pix[i];
            out[i] = RGB8::new(
                ((p.r as u32 * 255 + 32767) / 65535) as u8,
                ((p.g as u32 * 255 + 32767) / 65535) as u8,
                ((p.b as u32 * 255 + 32767) / 65535) as u8,
            );
        }
        return out;
    }

    // Floyd–Steinberg with scanline buffers, errors in 1/16 units on 8-bit domain.
    let mut err_cur = vec![0i32; w * 3];
    let mut err_nxt = vec![0i32; w * 3];

    for y in 0..h {
        err_nxt.fill(0);

        for x in 0..w {
            let idx = y * w + x;
            let p = pix[idx];

            let base_r = ((p.r as u32 * 255 + 32767) / 65535) as i32;
            let base_g = ((p.g as u32 * 255 + 32767) / 65535) as i32;
            let base_b = ((p.b as u32 * 255 + 32767) / 65535) as i32;

            let eoff = x * 3;

            let rr = base_r + (err_cur[eoff + 0] / 16);
            let gg = base_g + (err_cur[eoff + 1] / 16);
            let bb = base_b + (err_cur[eoff + 2] / 16);

            let qr = rr.clamp(0, 255);
            let qg = gg.clamp(0, 255);
            let qb = bb.clamp(0, 255);

            out[idx] = RGB8::new(qr as u8, qg as u8, qb as u8);

            // quantization error (scaled *16)
            let er = (rr - qr) * 16;
            let eg = (gg - qg) * 16;
            let eb = (bb - qb) * 16;

            // right (7/16)
            if x + 1 < w {
                err_cur[(x + 1) * 3 + 0] += (er * 7) / 16;
                err_cur[(x + 1) * 3 + 1] += (eg * 7) / 16;
                err_cur[(x + 1) * 3 + 2] += (eb * 7) / 16;
            }
            // next row
            if y + 1 < h {
                // down-left (3/16)
                if x > 0 {
                    err_nxt[(x - 1) * 3 + 0] += (er * 3) / 16;
                    err_nxt[(x - 1) * 3 + 1] += (eg * 3) / 16;
                    err_nxt[(x - 1) * 3 + 2] += (eb * 3) / 16;
                }
                // down (5/16)
                err_nxt[x * 3 + 0] += (er * 5) / 16;
                err_nxt[x * 3 + 1] += (eg * 5) / 16;
                err_nxt[x * 3 + 2] += (eb * 5) / 16;

                // down-right (1/16)
                if x + 1 < w {
                    err_nxt[(x + 1) * 3 + 0] += (er * 1) / 16;
                    err_nxt[(x + 1) * 3 + 1] += (eg * 1) / 16;
                    err_nxt[(x + 1) * 3 + 2] += (eb * 1) / 16;
                }
            }
        }

        std::mem::swap(&mut err_cur, &mut err_nxt);
        err_cur.fill(0);
    }

    out
}

// ---------------- Processing (sRGB working, Lab L sharpen, chroma stable) ----------------

fn enhance_srgb_rgb16(
    wp: WhitePoint,
    de_mode: DeltaEMode,
    de_max: f32,
    w: u32,
    h: u32,
    srgb: &[RGB16],
    scale: u32,
    shading_on: bool,
    p: &Params,
) -> Result<(u32, u32, Vec<RGB16>)> {
    match wp {
        WhitePoint::D65 => enhance_srgb_rgb16_d65(de_mode, de_max, w, h, srgb, scale, shading_on, p),
        WhitePoint::D50 => enhance_srgb_rgb16_d50(de_mode, de_max, w, h, srgb, scale, shading_on, p),
    }
}

fn enhance_srgb_rgb16_d65(
    de_mode: DeltaEMode,
    de_max: f32,
    w: u32,
    h: u32,
    srgb: &[RGB16],
    scale: u32,
    shading_on: bool,
    p: &Params,
) -> Result<(u32, u32, Vec<RGB16>)> {
    if srgb.len() != (w as usize) * (h as usize) {
        bail!("Pixel buffer size mismatch");
    }

    let mut l = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    let mut a = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    let mut b = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);

    for yy in 0..h {
        for xx in 0..w {
            let i = (yy as usize) * (w as usize) + (xx as usize);
            let p16 = srgb[i];
            let lab = rgb16_srgb_to_lab_f32_d65(p16);
            l.put_pixel(xx, yy, image::Luma([lab.l]));
            a.put_pixel(xx, yy, image::Luma([lab.a]));
            b.put_pixel(xx, yy, image::Luma([lab.b]));
        }
    }

    if shading_on {
        l = shading_correct_l(&l, p.shading_sigma, p.shading_strength);
        clamp_luma_inplace(&mut l, 0.0, 100.0);
    }

    let a_dn = if p.chroma_sigma > 0.0 { gaussian_blur_f32(&a, p.chroma_sigma) } else { a };
    let b_dn = if p.chroma_sigma > 0.0 { gaussian_blur_f32(&b, p.chroma_sigma) } else { b };

    let base = gaussian_blur_f32(&l, p.base_sigma);
    let mut detail = diff_luma(&l, &base);

    if p.detail_sigma > 0.0 {
        detail = gaussian_blur_f32(&detail, p.detail_sigma);
    }

    let l_base_up = resize_l_plane_u16(&base, scale);
    let mut l_detail_up = resize_l_plane_u16(&detail, scale);
    let a_up = resize_ab_plane_u16(&a_dn, scale);
    let b_up = resize_ab_plane_u16(&b_dn, scale);

    // "Before" L (pre-sharpen) for ΔE limiter
    let mut l_before = add_luma(&l_base_up, &l_detail_up);
    clamp_luma_inplace(&mut l_before, 0.0, 100.0);

    let mask = edge_mask_l(&l_before, p.edge_thresh, p.edge_softness);

    l_detail_up = masked_limited_unsharp_l(
        &l_detail_up,
        &mask,
        p.sharpen_strength,
        p.sharpen_blur_sigma,
        p.limiter_16bit,
    );

    let mut l_after = add_luma(&l_base_up, &l_detail_up);
    clamp_luma_inplace(&mut l_after, 0.0, 100.0);

    // ΔE limiter: blend L towards before if ΔE too large
    if !matches!(de_mode, DeltaEMode::None) && de_max > 0.0 {
        let (ow, oh) = l_after.dimensions();
        for yy in 0..oh {
            for xx in 0..ow {
                let l0 = l_before.get_pixel(xx, yy).0[0];
                let l1 = l_after.get_pixel(xx, yy).0[0];
                let aa = a_up.get_pixel(xx, yy).0[0];
                let bb = b_up.get_pixel(xx, yy).0[0];

                let de = delta_e(de_mode, (l0, aa, bb), (l1, aa, bb));
                if de > de_max && de > 1e-6 {
                    let k = (de_max / de).clamp(0.0, 1.0);
                    let lf = lerp(l0, l1, k);
                    l_after.put_pixel(xx, yy, image::Luma([lf]));
                }
            }
        }
    }

    let (ow, oh) = l_after.dimensions();
    let mut out = vec![RGB16::new(0, 0, 0); (ow as usize) * (oh as usize)];

    for yy in 0..oh {
        for xx in 0..ow {
            let lab = LabD65::new(
                l_after.get_pixel(xx, yy).0[0].clamp(0.0, 100.0),
                a_up.get_pixel(xx, yy).0[0].clamp(-128.0, 128.0),
                b_up.get_pixel(xx, yy).0[0].clamp(-128.0, 128.0),
            );
            let rgb = lab_to_rgb16_srgb_d65(lab);
            out[(yy as usize) * (ow as usize) + (xx as usize)] = rgb;
        }
    }

    Ok((ow, oh, out))
}

fn enhance_srgb_rgb16_d50(
    de_mode: DeltaEMode,
    de_max: f32,
    w: u32,
    h: u32,
    srgb: &[RGB16],
    scale: u32,
    shading_on: bool,
    p: &Params,
) -> Result<(u32, u32, Vec<RGB16>)> {
    if srgb.len() != (w as usize) * (h as usize) {
        bail!("Pixel buffer size mismatch");
    }

    let mut l = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    let mut a = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    let mut b = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);

    for yy in 0..h {
        for xx in 0..w {
            let i = (yy as usize) * (w as usize) + (xx as usize);
            let p16 = srgb[i];
            let lab = rgb16_srgb_to_lab_f32_d50(p16);
            l.put_pixel(xx, yy, image::Luma([lab.l]));
            a.put_pixel(xx, yy, image::Luma([lab.a]));
            b.put_pixel(xx, yy, image::Luma([lab.b]));
        }
    }

    if shading_on {
        l = shading_correct_l(&l, p.shading_sigma, p.shading_strength);
        clamp_luma_inplace(&mut l, 0.0, 100.0);
    }

    let a_dn = if p.chroma_sigma > 0.0 { gaussian_blur_f32(&a, p.chroma_sigma) } else { a };
    let b_dn = if p.chroma_sigma > 0.0 { gaussian_blur_f32(&b, p.chroma_sigma) } else { b };

    let base = gaussian_blur_f32(&l, p.base_sigma);
    let mut detail = diff_luma(&l, &base);

    if p.detail_sigma > 0.0 {
        detail = gaussian_blur_f32(&detail, p.detail_sigma);
    }

    let l_base_up = resize_l_plane_u16(&base, scale);
    let mut l_detail_up = resize_l_plane_u16(&detail, scale);
    let a_up = resize_ab_plane_u16(&a_dn, scale);
    let b_up = resize_ab_plane_u16(&b_dn, scale);

    let mut l_before = add_luma(&l_base_up, &l_detail_up);
    clamp_luma_inplace(&mut l_before, 0.0, 100.0);

    let mask = edge_mask_l(&l_before, p.edge_thresh, p.edge_softness);

    l_detail_up = masked_limited_unsharp_l(
        &l_detail_up,
        &mask,
        p.sharpen_strength,
        p.sharpen_blur_sigma,
        p.limiter_16bit,
    );

    let mut l_after = add_luma(&l_base_up, &l_detail_up);
    clamp_luma_inplace(&mut l_after, 0.0, 100.0);

    if !matches!(de_mode, DeltaEMode::None) && de_max > 0.0 {
        let (ow, oh) = l_after.dimensions();
        for yy in 0..oh {
            for xx in 0..ow {
                let l0 = l_before.get_pixel(xx, yy).0[0];
                let l1 = l_after.get_pixel(xx, yy).0[0];
                let aa = a_up.get_pixel(xx, yy).0[0];
                let bb = b_up.get_pixel(xx, yy).0[0];

                let de = delta_e(de_mode, (l0, aa, bb), (l1, aa, bb));
                if de > de_max && de > 1e-6 {
                    let k = (de_max / de).clamp(0.0, 1.0);
                    let lf = lerp(l0, l1, k);
                    l_after.put_pixel(xx, yy, image::Luma([lf]));
                }
            }
        }
    }

    let (ow, oh) = l_after.dimensions();
    let mut out = vec![RGB16::new(0, 0, 0); (ow as usize) * (oh as usize)];

    for yy in 0..oh {
        for xx in 0..ow {
            let lab = LabD50::new(
                l_after.get_pixel(xx, yy).0[0].clamp(0.0, 100.0),
                a_up.get_pixel(xx, yy).0[0].clamp(-128.0, 128.0),
                b_up.get_pixel(xx, yy).0[0].clamp(-128.0, 128.0),
            );
            let rgb = lab_to_rgb16_srgb_d50(lab);
            out[(yy as usize) * (ow as usize) + (xx as usize)] = rgb;
        }
    }

    Ok((ow, oh, out))
}

fn rgb16_srgb_to_lab_f32_d65(p: RGB16) -> LabD65 {
    // sRGB is D65-based; LinSrgb -> Lab(D65) is direct.
    let srgb = Srgb::new(
        p.r as f32 / 65535.0,
        p.g as f32 / 65535.0,
        p.b as f32 / 65535.0,
    );
    let lin: LinSrgb<f32> = srgb.into_linear();
    lin.into_color()
}

fn rgb16_srgb_to_lab_f32_d50(p: RGB16) -> LabD50 {
    // sRGB is D65-based; go via XYZ(D65), adapt to XYZ(D50) (Bradford), then Lab(D50).
    let srgb = Srgb::new(
        p.r as f32 / 65535.0,
        p.g as f32 / 65535.0,
        p.b as f32 / 65535.0,
    );
    let lin: LinSrgb<f32> = srgb.into_linear();

    let xyz_d65: Xyz<D65, f32> = lin.into_color();
    let xyz_d50: Xyz<D50, f32> = xyz_d65.adapt_into_using(Method::Bradford);
    xyz_d50.into_color()
}

fn lab_to_rgb16_srgb_d65(lab: LabD65) -> RGB16 {
    // Lab(D65) -> LinSrgb(D65) -> sRGB
    let lin: LinSrgb<f32> = lab.into_color();
    let srgb: Srgb<f32> = Srgb::from_linear(lin);

    let r = (srgb.red.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let g = (srgb.green.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let b = (srgb.blue.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    RGB16::new(r, g, b)
}

fn lab_to_rgb16_srgb_d50(lab: LabD50) -> RGB16 {
    // Lab(D50) -> XYZ(D50) -> adapt to XYZ(D65) (Bradford) -> LinSrgb(D65) -> sRGB
    let xyz_d50: Xyz<D50, f32> = lab.into_color();
    let xyz_d65: Xyz<D65, f32> = xyz_d50.adapt_into_using(Method::Bradford);

    let lin: LinSrgb<f32> = xyz_d65.into_color();
    let srgb: Srgb<f32> = Srgb::from_linear(lin);

    let r = (srgb.red.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let g = (srgb.green.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    let b = (srgb.blue.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
    RGB16::new(r, g, b)
}

fn resize_l_plane_u16(
    src: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    scale: u32,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    if scale == 1 {
        return src.clone();
    }
    let (w, h) = src.dimensions();
    let nw = w.saturating_mul(scale);
    let nh = h.saturating_mul(scale);

    let mut tmp16 = ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
    for (x, y, p) in src.enumerate_pixels() {
        let v = (p.0[0].clamp(0.0, 100.0) / 100.0 * 65535.0 + 0.5) as u16;
        tmp16.put_pixel(x, y, image::Luma([v]));
    }

    let resized16 = image::imageops::resize(&tmp16, nw, nh, FilterType::Lanczos3);

    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(nw, nh);
    for (x, y, p) in resized16.enumerate_pixels() {
        out.put_pixel(x, y, image::Luma([p.0[0] as f32 / 65535.0 * 100.0]));
    }
    out
}

fn resize_ab_plane_u16(
    src: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    scale: u32,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    if scale == 1 {
        return src.clone();
    }
    let (w, h) = src.dimensions();
    let nw = w.saturating_mul(scale);
    let nh = h.saturating_mul(scale);

    let mut tmp16 = ImageBuffer::<image::Luma<u16>, Vec<u16>>::new(w, h);
    for (x, y, p) in src.enumerate_pixels() {
        let v = ((p.0[0].clamp(-128.0, 128.0) + 128.0) / 256.0 * 65535.0 + 0.5) as u16;
        tmp16.put_pixel(x, y, image::Luma([v]));
    }

    let resized16 = image::imageops::resize(&tmp16, nw, nh, FilterType::Lanczos3);

    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(nw, nh);
    for (x, y, p) in resized16.enumerate_pixels() {
        out.put_pixel(x, y, image::Luma([p.0[0] as f32 / 65535.0 * 256.0 - 128.0]));
    }
    out
}

fn diff_luma(
    a: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    b: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    let (w, h) = a.dimensions();
    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    for yy in 0..h {
        for xx in 0..w {
            out.put_pixel(
                xx,
                yy,
                image::Luma([a.get_pixel(xx, yy).0[0] - b.get_pixel(xx, yy).0[0]]),
            );
        }
    }
    out
}

fn add_luma(
    a: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    b: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    let (w, h) = a.dimensions();
    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    for yy in 0..h {
        for xx in 0..w {
            out.put_pixel(
                xx,
                yy,
                image::Luma([a.get_pixel(xx, yy).0[0] + b.get_pixel(xx, yy).0[0]]),
            );
        }
    }
    out
}

fn clamp_luma_inplace(img: &mut ImageBuffer<image::Luma<f32>, Vec<f32>>, lo: f32, hi: f32) {
    for p in img.pixels_mut() {
        p.0[0] = p.0[0].clamp(lo, hi);
    }
}

fn shading_correct_l(
    l: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    sigma: f32,
    strength: f32,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    let bg = gaussian_blur_f32(l, sigma);

    let mut sum = 0.0f32;
    let mut n = 0.0f32;
    for p in bg.pixels() {
        sum += p.0[0];
        n += 1.0;
    }
    let mean_bg = (sum / n).max(1e-3);

    let eps = 1e-3f32;
    let (w, h) = l.dimensions();
    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);

    for yy in 0..h {
        for xx in 0..w {
            let v = l.get_pixel(xx, yy).0[0];
            let b = bg.get_pixel(xx, yy).0[0];
            let corr = v * (mean_bg / (b + eps));
            let blended = v * (1.0 - strength) + corr * strength;
            out.put_pixel(xx, yy, image::Luma([blended]));
        }
    }
    out
}

fn edge_mask_l(
    l: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    thresh: f32,
    softness: f32,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    let (w, h) = l.dimensions();

    let mut l8 = ImageBuffer::<image::Luma<u8>, Vec<u8>>::new(w, h);
    for (x, yy, p) in l.enumerate_pixels() {
        let v = (p.0[0].clamp(0.0, 100.0) / 100.0 * 255.0 + 0.5) as u8;
        l8.put_pixel(x, yy, image::Luma([v]));
    }

    let gx = horizontal_sobel(&l8);
    let gy = vertical_sobel(&l8);

    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);

    for yy in 0..h {
        for xx in 0..w {
            let sx = gx.get_pixel(xx, yy).0[0] as f32;
            let sy = gy.get_pixel(xx, yy).0[0] as f32;

            let mag = ((sx * sx + sy * sy).sqrt() / 1500.0).clamp(0.0, 1.0);

            let t0 = thresh;
            let t1 = (thresh + softness).min(1.0);
            let m = if mag <= t0 {
                0.0
            } else if mag >= t1 {
                1.0
            } else {
                let u = (mag - t0) / (t1 - t0);
                u * u * (3.0 - 2.0 * u)
            };

            out.put_pixel(xx, yy, image::Luma([m]));
        }
    }
    out
}

fn masked_limited_unsharp_l(
    detail: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    mask: &ImageBuffer<image::Luma<f32>, Vec<f32>>,
    strength: f32,
    blur_sigma: f32,
    limiter_16bit: f32,
) -> ImageBuffer<image::Luma<f32>, Vec<f32>> {
    if strength <= 0.0 {
        return detail.clone();
    }

    let blur = gaussian_blur_f32(detail, blur_sigma);
    let (w, h) = detail.dimensions();

    let limiter_l = (limiter_16bit / 65535.0) * 100.0;

    let mut out = ImageBuffer::<image::Luma<f32>, Vec<f32>>::new(w, h);
    for yy in 0..h {
        for xx in 0..w {
            let d = detail.get_pixel(xx, yy).0[0];
            let b = blur.get_pixel(xx, yy).0[0];
            let m = mask.get_pixel(xx, yy).0[0];

            let hp = d - b;
            let mut delta = strength * hp;
            delta = delta.clamp(-limiter_l, limiter_l);

            out.put_pixel(xx, yy, image::Luma([d + delta * m]));
        }
    }
    out
}

// ---------------- Main flow ----------------

fn main() -> Result<()> {
    let cli = Cli::parse();
    fs::create_dir_all(&cli.output_dir).context("create output_dir")?;

    if cli.scale != 1 && cli.scale != 2 {
        bail!("Only --scale 1 or --scale 2 is supported.");
    }
    if cli.delta_e_max < 0.0 {
        bail!("--delta-e-max must be >= 0");
    }

    if cli.input.is_dir() {
        let files: Vec<PathBuf> = WalkDir::new(&cli.input)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.into_path())
            .filter(|p| is_tiff(p))
            .collect();

        if files.is_empty() {
            bail!("No TIFF files found in {}", cli.input.display());
        }

        if cli.parallel {
            files.par_iter().try_for_each(|p| process_one(p, &cli))?;
        } else {
            for p in files {
                process_one(&p, &cli)?;
            }
        }
    } else {
        if !is_tiff(&cli.input) {
            bail!("Input is not a TIFF: {}", cli.input.display());
        }
        process_one(&cli.input, &cli)?;
    }

    Ok(())
}

fn process_one(path: &Path, cli: &Cli) -> Result<()> {
    let out_path = cli.output_dir.join(path.file_name().unwrap()).with_extension("tif");
    if out_path.exists() && !cli.force {
        return Ok(());
    }

    let mut meta = read_tiff_meta(path).with_context(|| format!("read_tiff_meta {}", path.display()))?;
    let (w, h, mut pix) = read_tiff_rgb16(path).with_context(|| format!("decode {}", path.display()))?;

    let srgb_prof = Profile::new_srgb();
    let in_prof = pick_input_profile(&meta).with_context(|| "pick_input_profile")?;
    let out_prof_opt = pick_output_profile(cli, &meta).with_context(|| "pick_output_profile")?;

    let intent: Intent = cli.intent.into();
    let params = Params::for_preset(cli.preset);

    // Input -> sRGB (working)
    if !cli.no_icc {
        let to_srgb = build_transform_rgb16(&in_prof, &srgb_prof, intent, cli.bpc)
            .with_context(|| "build in->sRGB transform")?;
        to_srgb.transform_in_place(&mut pix);
    }

    let (ow, oh, mut out_pix) = enhance_srgb_rgb16(
        cli.white_point,
        cli.delta_e,
        cli.delta_e_max,
        w,
        h,
        &pix,
        cli.scale,
        cli.shading,
        &params,
    )
    .with_context(|| format!("enhance {}", path.display()))?;

    // sRGB -> output profile (if requested), and decide ICC bytes to embed
    let embed_icc: Option<Vec<u8>> = if let Some(out_prof) = out_prof_opt.as_ref() {
        if !cli.no_icc {
            let from_srgb = build_transform_rgb16(&srgb_prof, out_prof, intent, cli.bpc)
                .with_context(|| "build sRGB->out transform")?;
            from_srgb.transform_in_place(&mut out_pix);
        }

        match cli.icc_mode {
            IccMode::None => None,
            _ => match out_prof.icc() {
                Ok(bytes) => Some(bytes),
                Err(e) => {
                    eprintln!("Warning: cannot export output ICC for {}: {}", out_path.display(), e);
                    None
                }
            },
        }
    } else {
        None
    };

    // DPI scaling when upscaling (keep physical size same)
    if cli.dpi_scale && cli.scale > 1 {
        let s = cli.scale;
        if let Some(x) = meta.x_res.as_mut() {
            x.n = x.n.saturating_mul(s);
        }
        if let Some(y) = meta.y_res.as_mut() {
            y.n = y.n.saturating_mul(s);
        }
    }

    match cli.out_depth {
        OutDepth::B16 => {
            write_tiff_rgb16(&out_path, ow, oh, &out_pix, &meta, embed_icc.as_deref())
                .with_context(|| format!("write {}", out_path.display()))?;
        }
        OutDepth::B8 => {
            let out8 = quantize_rgb16_to_rgb8(&out_pix, ow, oh, cli.dither);
            write_tiff_rgb8(&out_path, ow, oh, &out8, &meta, embed_icc.as_deref())
                .with_context(|| format!("write {}", out_path.display()))?;
        }
    }

    Ok(())
}
