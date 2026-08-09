import { useMutation, useQuery } from "@tanstack/react-query";
import { uploadRankingFiles } from "../services/upload";
import { getRankingStatus } from "../services/status";
import { getCandidate, getRankingResults } from "../services/results";
import { downloadRankingFile } from "../services/download";
import { getLibraryCandidates, getLibraryJobDescriptions, startLibraryRanking } from "../services/library";
import type { ResultsParams } from "../types/api";

export const useUploadRanking = () => useMutation({ mutationFn: uploadRankingFiles });
export const useRankingStatus = (jobId?: string) => useQuery({ queryKey: ["ranking-status", jobId], queryFn: () => getRankingStatus(jobId!), enabled: Boolean(jobId), refetchInterval: (query) => query.state.data?.state === "processing" ? 2500 : false });
export const useRankingResults = (jobId: string | undefined, params: ResultsParams) => useQuery({ queryKey: ["ranking-results", jobId, params], queryFn: () => getRankingResults(jobId!, params), enabled: Boolean(jobId), placeholderData: (previous) => previous });
export const useCandidate = (jobId?: string, candidateId?: string) => useQuery({ queryKey: ["candidate", jobId, candidateId], queryFn: () => getCandidate(jobId!, candidateId!), enabled: Boolean(jobId && candidateId) });
export const useDownloadRanking = () => useMutation({ mutationFn: ({ jobId, kind }: { jobId: string; kind: "submission" | "audit" | "all" }) => downloadRankingFile(jobId, kind) });
export const useDatasetLibrary = () => ({ candidates: useQuery({ queryKey: ["library-candidates"], queryFn: getLibraryCandidates }), jobDescriptions: useQuery({ queryKey: ["library-job-descriptions"], queryFn: getLibraryJobDescriptions }) });
export const useStartLibraryRanking = () => useMutation({ mutationFn: startLibraryRanking });
