import { api } from "./api";
import type { DownloadKind } from "../types/api";

export async function downloadRankingFile(jobId: string, kind: DownloadKind): Promise<Blob> {
  const response = await api.get<Blob>(`/rankings/${jobId}/downloads/${kind}`, { responseType: "blob" });
  return response.data;
}
