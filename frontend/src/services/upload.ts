import { api } from "./api";
import type { UploadResponse } from "../types/api";

export async function uploadRankingFiles(files: { candidates?: File; jobDescription?: File; metadata?: File }): Promise<UploadResponse> {
  if (!files.candidates || !files.jobDescription) {
    throw new Error("Candidates dataset and job description are required.");
  }
  const formData = new FormData();
  formData.append("candidates", files.candidates);
  formData.append("jobDescription", files.jobDescription);
  if (files.metadata) formData.append("metadata", files.metadata);
  // Candidate datasets can be hundreds of MB; let the browser wait for the
  // upload to finish instead of applying the normal API-request timeout.
  const response = await api.post<UploadResponse>("/rankings", formData, { timeout: 0 });
  return response.data;
}
