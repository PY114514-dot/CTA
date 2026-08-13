import type { CandidateCurvePoint } from "../components/nav/CandidateCurveOverlay";

/** Aligns a traced candidate vertically to pixels matching the user-picked curve colour. */
export async function autoFitCurveToSelectedLine(
  sourceUrl: string,
  curve: CandidateCurvePoint[],
  colorHex: string,
): Promise<{ curve: CandidateCurvePoint[]; matched: number; medianResidual: number }> {
  if (curve.length < 3) throw new Error("候选曲线点不足，无法自动贴合");
  const image = new Image();
  image.src = sourceUrl;
  await new Promise<void>((resolve, reject) => { image.onload = () => resolve(); image.onerror = () => reject(new Error("原图无法读取")); });
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("浏览器不支持图像贴合");
  context.drawImage(image, 0, 0);
  const { data } = context.getImageData(0, 0, canvas.width, canvas.height);
  const width = canvas.width;
  const height = canvas.height;
  const matches: Array<{ source: number; target: number }> = [];
  const scanRadius = Math.max(10, Math.round(height * 0.16));
  const token = colorHex.replace("#", "");
  const target = {
    r: Number.parseInt(token.slice(0, 2), 16),
    g: Number.parseInt(token.slice(2, 4), 16),
    b: Number.parseInt(token.slice(4, 6), 16),
  };
  if (![target.r, target.g, target.b].every(Number.isFinite)) throw new Error("请选择有效的曲线颜色");

  for (let index = 0; index < curve.length; index += Math.max(1, Math.floor(curve.length / 90))) {
    const point = curve[index]!;
    const x = Math.round(point.x * (width - 1));
    const expected = Math.round(point.y * (height - 1));
    const ys: number[] = [];
    for (let dx = -2; dx <= 2; dx += 1) {
      const px = Math.max(0, Math.min(width - 1, x + dx));
      for (let y = Math.max(0, expected - scanRadius); y <= Math.min(height - 1, expected + scanRadius); y += 1) {
        const offset = (y * width + px) * 4;
        const r = data[offset] ?? 0; const g = data[offset + 1] ?? 0; const b = data[offset + 2] ?? 0;
        if (Math.hypot(r - target.r, g - target.g, b - target.b) <= 76) ys.push(y);
      }
    }
    if (ys.length) {
      ys.sort((a, b) => a - b);
      matches.push({ source: point.y, target: ys[Math.floor(ys.length / 2)]! / Math.max(height - 1, 1) });
    }
  }
  if (matches.length < 6) throw new Error("未能在原图中稳定定位所选净值线，请改用三个锚点微调");
  const meanSource = matches.reduce((sum, item) => sum + item.source, 0) / matches.length;
  const meanTarget = matches.reduce((sum, item) => sum + item.target, 0) / matches.length;
  const denominator = matches.reduce((sum, item) => sum + (item.source - meanSource) ** 2, 0);
  const rawScale = denominator > 1e-8
    ? matches.reduce((sum, item) => sum + (item.source - meanSource) * (item.target - meanTarget), 0) / denominator
    : 1;
  const scale = Math.max(0.65, Math.min(1.35, rawScale));
  const offset = meanTarget - scale * meanSource;
  const fitted = curve.map((point) => ({ ...point, y: Math.max(0, Math.min(1, point.y * scale + offset)) }));
  const residuals = matches.map((item) => Math.abs(item.target - (item.source * scale + offset))).sort((a, b) => a - b);
  return { curve: fitted, matched: matches.length, medianResidual: residuals[Math.floor(residuals.length / 2)]! };
}
