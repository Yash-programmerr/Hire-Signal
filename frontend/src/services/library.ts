import { api } from "./api";
import type { UploadResponse } from "../types/api";

export interface LibraryFile {
  id: string;
  name: string;
  extension: string;
  sizeBytes: number;
}

type LibraryResponse = { items: LibraryFile[] };

export async function getLibraryCandidates(): Promise<LibraryFile[]> {
  const response = await api.get<LibraryResponse>("/dataset-library/candidates");
  return response.data.items;
}

export async function getLibraryJobDescriptions(): Promise<LibraryFile[]> {
  const response = await api.get<LibraryResponse>("/dataset-library/job-descriptions");
  return response.data.items;
}

export async function startLibraryRanking(input: { candidateId: string; jobDescriptionId: string }): Promise<UploadResponse> {
  const response = await api.post<UploadResponse>("/rankings/from-library", input);
  return response.data;
}
