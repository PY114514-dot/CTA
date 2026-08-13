/** Infer reviewable product identity hints from a material filename.
 * A filename is evidence, not a confirmed identity: callers must keep the
 * candidate editable and label it as such.
 */
export function inferIdentityFromFilename(filename: string): { productName: string; managerName?: string } {
  const stem = filename.replace(/\.[^.]+$/, "").replace(/[\s_-]*(净值|单位净值|产品净值|nav|净值表|周报|月报|日报)$/i, "").trim();
  const parts = stem.split(/[\s_—–-]+/).filter(Boolean);
  const managerIndex = parts.findIndex((part) => /投资|资产|基金|资本|期货|管理/.test(part));
  if (managerIndex >= 0 && parts.length > 1) {
    const managerName = parts[managerIndex];
    const productName = parts.filter((_, index) => index !== managerIndex).join(" ");
    return { productName: productName || stem, managerName };
  }
  return { productName: stem || filename };
}
