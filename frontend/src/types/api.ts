export type UploadResponse = { jobId: string };
export type RankingState = "queued" | "processing" | "completed" | "failed";
export interface RankingStatus { state: RankingState; progress: number; currentStep: string; error?: string; }
export interface RankingSummary { candidatesUploaded: number; candidatesRanked: number; filteredCandidates: number; averageMatchScore: number; processingTimeSeconds: number; }
export interface RankingCandidate { id: string; name: string; rank: number; score: number; matchPercentage: number; experience: string; }
export interface PageInfo { page: number; pageSize: number; total: number; totalPages: number; }
export interface RankingResults { summary: RankingSummary; candidates: RankingCandidate[]; page: PageInfo; }
export interface CandidateDetail extends RankingCandidate { matchedSkills?: string[]; education?: string; reasonCodes?: string[]; rankingExplanation?: string; timeline?: Array<{ title: string; company?: string; dateRange?: string; description?: string }>; }
export type DownloadKind = "submission" | "audit" | "all";
export interface ResultsParams { page: number; pageSize: number; search?: string; sort?: string; direction?: "asc" | "desc"; }
