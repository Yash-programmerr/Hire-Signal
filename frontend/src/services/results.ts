import { api } from "./api";
import type { CandidateDetail, RankingResults, ResultsParams } from "../types/api";

export async function getRankingResults(jobId: string, params: ResultsParams): Promise<RankingResults> {
  const response = await api.get<RankingResults>(`/rankings/${jobId}/results`, { params });
  return response.data;
}

export async function getCandidate(jobId: string, candidateId: string): Promise<CandidateDetail> {
  const response = await api.get<CandidateDetail>(`/rankings/${jobId}/candidates/${candidateId}`);
  return response.data;
}
