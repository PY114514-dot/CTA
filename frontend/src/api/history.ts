/**
 * History domain — research archive CRUD operations.
 */

import type {
  HistoryRecordPayload,
  HistoryRecordResponse,
} from "./types";
import { API_BASE_URL, errorMessage, requestJson } from "./client";

/** Fetch all history records from the backend (newest first, max 50). */
export async function listHistoryRecords(): Promise<HistoryRecordResponse[]> {
  return requestJson("/history", "加载研究档案失败");
}

/** Save a single history record to the backend. */
export async function createHistoryRecord(record: HistoryRecordPayload): Promise<HistoryRecordResponse> {
  const response = await fetch(`${API_BASE_URL}/history`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "保存研究档案失败"));
  return response.json() as Promise<HistoryRecordResponse>;
}

/** Bulk-sync localStorage records to the backend. */
export async function syncHistoryRecords(records: HistoryRecordPayload[]): Promise<HistoryRecordResponse[]> {
  const response = await fetch(`${API_BASE_URL}/history/sync`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ records }),
  });
  if (!response.ok) throw new Error(await errorMessage(response, "同步研究档案失败"));
  return response.json() as Promise<HistoryRecordResponse[]>;
}

/** Delete a single history record. */
export async function deleteHistoryRecord(id: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/history/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response, "删除研究档案失败"));
}

/** Clear all history records. */
export async function clearHistoryRecords(): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/history`, { method: "DELETE" });
  if (!response.ok) throw new Error(await errorMessage(response, "清空研究档案失败"));
}
