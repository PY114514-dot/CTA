/** 对比结论快照的本地持久化（localStorage）。
 *
 * 快照冻结保存时刻的结论（异同摘要、关键指标、相关性），
 * 产品数据之后更新也不影响已保存的结论；「重新对比」则按
 * 保存的产品编号打开实时对比。
 */

export interface CompareSnapshotMetric {
  label: string;
  leftText: string;
  rightText: string;
  diffText: string;
  /** 保存时刻的占优侧；null 表示未达显著门槛或无法比较。 */
  better: "left" | "right" | null;
}

export interface CompareSnapshotRecord {
  id: string;
  savedAt: string;
  leftId: string;
  leftName: string;
  rightId: string;
  rightName: string;
  correlation: number | null;
  overlap: number;
  correlationNote: string | null;
  similarities: string[];
  differences: string[];
  metrics: CompareSnapshotMetric[];
}

const STORAGE_KEY = "cta_compare_snapshots";
const MAX_RECORDS = 50;

export function loadCompareSnapshots(): CompareSnapshotRecord[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CompareSnapshotRecord[]) : [];
  } catch {
    return [];
  }
}

function persist(records: CompareSnapshotRecord[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records));
  } catch {
    // 存储不可用（隐私模式或配额满）时静默失败，快照仅本次会话有效。
  }
}

export function saveCompareSnapshot(record: CompareSnapshotRecord): CompareSnapshotRecord[] {
  const records = [record, ...loadCompareSnapshots()].slice(0, MAX_RECORDS);
  persist(records);
  return records;
}

export function deleteCompareSnapshot(id: string): CompareSnapshotRecord[] {
  const records = loadCompareSnapshots().filter((record) => record.id !== id);
  persist(records);
  return records;
}
