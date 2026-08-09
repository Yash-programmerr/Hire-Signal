import { api } from "./api";
import type { RankingStatus } from "../types/api";

export async function getRankingStatus(jobId: string): Promise<RankingStatus> {
  const response = await api.get<RankingStatus>(`/rankings/${jobId}/status`);
  return response.data;
}
