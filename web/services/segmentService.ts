// segmentService.ts - Service for on-device object detection simulation
import axios, { AxiosInstance, AxiosResponse } from "axios";

const segmentApi: AxiosInstance = axios.create({
  // Use Vite env var VITE_SEGMENT_URL. Falls back to localhost:8000.
  baseURL: (import.meta as any).env?.VITE_SEGMENT_URL ?? "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
  timeout: 30_000, // 30 seconds for segmentation
});

export interface SegmentMask {
  index: number;
  path: string;
  png_base64: string;
}

export interface SegmentResponse {
  count: number;
  masks: SegmentMask[];
}

export type SegmentResult = {
  success: boolean;
  data?: SegmentResponse;
  error?: string;
};

/**
 * Send image to segmentation service and get back object masks
 * @param imageName - The name of the image on the backend
 */
export async function segment(imageName: string): Promise<SegmentResult> {
  try {
    const resp: AxiosResponse<SegmentResponse> = await segmentApi.post(
      "/segment",
      { image_name: imageName }
    );
    return { success: true, data: resp.data };
  } catch (err: any) {
    console.error("Error in segment API call:", err);
    return {
      success: false,
      error: err.response?.data?.error ?? err.message ?? "Segmentation failed",
    };
  }
}
