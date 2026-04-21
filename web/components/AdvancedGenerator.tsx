import React, { useState, useRef, useEffect } from "react";
import {
  Upload,
  X,
  Terminal,
  Image as ImageIcon,
  Sparkles,
  AlertCircle,
  ArrowRight,
  Download,
} from "lucide-react";
import SystemUsageChart from "./SystemUsageChart";
import * as api from "../services/features";

const AdvancedGenerator: React.FC = () => {
  // Main Image State
  const [mainImage, setMainImage] = useState<{
    data: string;
    mimeType: string;
  } | null>(null);

  // Reference Image State
  const [refImage, setRefImage] = useState<{
    data: string;
    mimeType: string;
  } | null>(null);

  // Prompt State
  const [prompt, setPrompt] = useState<string>("");

  // Selected Style
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null);

  // System State
  const [logs, setLogs] = useState<string[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [resultImage, setResultImage] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<Array<{
    timestamp: number;
    cpu: number;
    gpu: number;
    memory: number;
  }> | null>(null);
  const logsEndRef = useRef<HTMLDivElement>(null);

  const mainInputRef = useRef<HTMLInputElement>(null);
  const refInputRef = useRef<HTMLInputElement>(null);

  // Available styles
  const styles = ["FORMAL", "VOGUE", "GHIBLI"];

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const addLog = (message: string) => {
    setLogs((prev) => [
      ...prev,
      `[${new Date().toLocaleTimeString()}] ${message}`,
    ]);
  };

  const handleImageUpload = (
    e: React.ChangeEvent<HTMLInputElement>,
    isMain: boolean
  ) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (ev) => {
        const result = ev.target?.result as string;
        const base64Data = result.split(",")[1];
        const imageData = { data: base64Data, mimeType: file.type };

        if (isMain) {
          setMainImage(imageData);
          addLog(`Main image uploaded: ${file.name}`);
        } else {
          setRefImage(imageData);
          addLog(`Reference image uploaded: ${file.name}`);
        }
      };
      reader.readAsDataURL(file);
    }
  };

  const clearMainImage = () => {
    setMainImage(null);
    setResultImage(null);
    setMetrics(null);
    if (mainInputRef.current) mainInputRef.current.value = "";
    addLog("Main image cleared.");
  };

  const clearRefImage = () => {
    setRefImage(null);
    if (refInputRef.current) refInputRef.current.value = "";
    addLog("Reference image cleared.");
  };

  const downloadImage = async () => {
    const imageToDownload =
      resultImage ||
      (mainImage
        ? `data:${mainImage.mimeType};base64,${mainImage.data}`
        : null);
    if (!imageToDownload) {
      addLog("No image to download.");
      return;
    }

    try {
      const response = await fetch(imageToDownload);
      const blob = await response.blob();

      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `image_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      addLog("Image downloaded successfully.");
    } catch (error: any) {
      addLog(`Download failed: ${error.message}`);
    }
  };

  const executePipeline = async () => {
    if (!mainImage) {
      addLog("ERROR: Source image is required.");
      return;
    }

    setIsProcessing(true);
    setResultImage(null);
    addLog("Initializing stylize pipeline...");

    try {
      // Always use stylize API
      const promptToSend = prompt.trim() || undefined;
      const styleToSend = selectedStyle || undefined;
      const refImageToSend = refImage ? refImage.data : undefined;

      addLog("Sending to /stylize API...");
      if (promptToSend) addLog(`  - Prompt: "${promptToSend}"`);
      if (styleToSend) addLog(`  - Style: ${styleToSend}`);
      if (refImageToSend) addLog(`  - Reference image included`);

      const response = await api.stylize(
        mainImage.data,
        promptToSend,
        styleToSend,
        refImageToSend
      );

      // Handle response
      if (response && response.success && response.data) {
        addLog(`✓ Processing complete: ${response.data.message}`);

        // Extract metrics if available
        if (response.data.metrics) {
          setMetrics(response.data.metrics);
          addLog(`✓ Received ${response.data.metrics.length} metric samples`);
        }

        if (response.data.image) {
          const resultDataUrl = `data:image/png;base64,${response.data.image}`;
          setResultImage(resultDataUrl);
          addLog("✓ Result image received and displayed.");
        }
      } else {
        addLog(`✗ Processing failed: ${response?.error || "Unknown error"}`);
      }
    } catch (error: any) {
      addLog(`CRITICAL ERROR: ${error.message}`);
      console.error(error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex flex-row h-full gap-3">
      {/* LEFT COLUMN: Main Visuals + Metrics */}
      <div className="flex-1 flex flex-col min-w-0 gap-3">
        {/* Main Image Viewer */}
        <div className="h-[55vh] bg-zinc-900 border border-zinc-800 rounded-xl relative overflow-hidden flex items-center justify-center p-4 group">
          {!mainImage ? (
            <div
              onClick={() => mainInputRef.current?.click()}
              className="absolute inset-0 flex flex-col items-center justify-center cursor-pointer hover:bg-zinc-800/30 transition-colors"
            >
              <div className="w-20 h-20 rounded-full bg-zinc-800 border-2 border-dashed border-zinc-700 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <Upload className="w-8 h-8 text-zinc-500" />
              </div>
              <h3 className="text-zinc-400 font-medium">Upload Source Image</h3>
              <input
                type="file"
                ref={mainInputRef}
                onChange={(e) => handleImageUpload(e, true)}
                accept="image/*"
                className="hidden"
              />
            </div>
          ) : (
            <div className="relative w-full h-full flex items-center justify-center">
              <img
                src={
                  resultImage ||
                  `data:${mainImage.mimeType};base64,${mainImage.data}`
                }
                alt={resultImage ? "Result" : "Source"}
                className="max-w-full max-h-full object-contain shadow-2xl"
              />
              {resultImage && (
                <div className="absolute bottom-2 left-2 px-3 py-1 bg-green-500/90 text-white text-xs font-bold rounded-full backdrop-blur-sm z-30">
                  ✓ RESULT
                </div>
              )}
              <button
                onClick={clearMainImage}
                className="absolute top-2 left-2 p-2 bg-black/60 text-zinc-400 hover:text-red-400 rounded-lg backdrop-blur-md transition-colors z-40"
              >
                <X className="w-5 h-5" />
              </button>
              <button
                onClick={downloadImage}
                className="absolute bottom-2 right-2 p-2 bg-black/60 text-zinc-400 hover:text-green-400 rounded-lg backdrop-blur-md transition-colors z-40"
                title="Download image"
              >
                <Download className="w-5 h-5" />
              </button>
            </div>
          )}
        </div>

        {/* System Metrics */}
        <div className="flex-1 min-h-0">
          <SystemUsageChart metrics={metrics} />
        </div>
      </div>

      {/* RIGHT COLUMN: Context & Controls */}
      <div className="w-[400px] shrink-0 flex flex-col gap-3">
        {/* Configuration Panel - Matching SEMI height */}
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 flex flex-col gap-4 h-[55vh] overflow-y-auto scrollbar-thin">
          <div className="flex items-center gap-2 text-zinc-400 uppercase tracking-widest text-xs font-bold border-b border-zinc-800 pb-2">
            <Sparkles className="w-4 h-4" />
            Style Configuration
          </div>

          <p className="text-zinc-400 text-sm -mt-2">
            <span className="text-blue-300 font-medium">
              At least one option is required
            </span>
          </p>

          {/* Reference Image Input */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-zinc-300 flex justify-between items-center min-h-[20px]">
              <span>Reference Image (Optional)</span>
              {refImage && (
                <button
                  onClick={clearRefImage}
                  className="text-red-400 hover:text-red-300 text-[10px] uppercase border border-red-500/50 px-1.5 py-0.5 rounded hover:border-red-400 transition-colors leading-none"
                >
                  Remove
                </button>
              )}
            </div>

            {!refImage ? (
              <div
                onClick={() => refInputRef.current?.click()}
                className="h-24 w-full border border-dashed border-zinc-700 rounded-lg hover:bg-zinc-800/50 transition-colors cursor-pointer flex flex-col items-center justify-center gap-2"
              >
                <Upload className="w-5 h-5 text-zinc-600" />
                <span className="text-xs text-zinc-600">
                  Click to upload reference style/image
                </span>
                <input
                  type="file"
                  ref={refInputRef}
                  onChange={(e) => handleImageUpload(e, false)}
                  accept="image/*"
                  className="hidden"
                />
              </div>
            ) : (
              <div className="h-24 w-full bg-black rounded-lg border border-zinc-700 overflow-hidden relative group">
                <img
                  src={`data:${refImage.mimeType};base64,${refImage.data}`}
                  alt="Ref"
                  className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity"
                />
                <div className="absolute bottom-1 right-1 bg-black/70 px-2 py-0.5 rounded text-[10px] text-zinc-300">
                  REF
                </div>
              </div>
            )}
          </div>

          {/* Prompt Input */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-zinc-300">
              Text Prompt (Optional)
            </div>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe the style or transformation..."
              className="w-full h-24 bg-black border border-zinc-800 rounded-lg p-3 text-sm text-zinc-300 placeholder:text-zinc-700 resize-none focus:outline-none focus:border-blue-500/50 transition-colors"
            />
          </div>

          {/* Styles Section */}
          <div className="space-y-2">
            <div className="text-xs font-medium text-zinc-300">
              Styles (Optional)
            </div>
            <div className="flex gap-3 overflow-x-auto scrollbar-thin pb-2">
              {styles.map((style) => (
                <button
                  key={style}
                  onClick={() =>
                    setSelectedStyle(selectedStyle === style ? null : style)
                  }
                  className={`flex flex-col items-center gap-2 p-3 rounded-lg transition-all shrink-0 w-24 ${
                    selectedStyle === style
                      ? "bg-blue-600/20 border-2 border-blue-500 shadow-lg shadow-blue-500/20"
                      : "bg-zinc-800 border border-zinc-700 hover:border-blue-500"
                  }`}
                >
                  {/* Placeholder Icon/Shape */}
                  <div
                    className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                      selectedStyle === style
                        ? "bg-blue-500 text-white"
                        : "bg-zinc-700 text-zinc-500"
                    }`}
                  >
                    <Sparkles className="w-6 h-6" />
                  </div>
                  {/* Style Name */}
                  <span
                    className={`text-[10px] font-medium text-center leading-tight ${
                      selectedStyle === style
                        ? "text-blue-300"
                        : "text-zinc-400"
                    }`}
                  >
                    {style}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* Run Button - Matching SEMI button size */}
          <button
            onClick={executePipeline}
            disabled={isProcessing || !mainImage}
            className={`w-full py-3 rounded-xl font-bold uppercase tracking-wider transition-all duration-300 flex items-center justify-center gap-3 text-xs
                    ${
                      isProcessing || !mainImage
                        ? "bg-zinc-800 text-zinc-600 cursor-not-allowed"
                        : "bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg hover:shadow-emerald-900/40"
                    }`}
          >
            {isProcessing ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                <span>Processing...</span>
              </>
            ) : (
              <>
                <ArrowRight className="w-4 h-4" />
                <span>Generate</span>
              </>
            )}
          </button>

          {/* Validation Message */}
          {!mainImage && (
            <div className="flex items-center gap-2 text-xs text-amber-500/80 justify-center">
              <AlertCircle className="w-3 h-3" />
              <span>Source image required</span>
            </div>
          )}
        </div>

        {/* Logs Console */}
        <div className="flex-1 min-h-0 bg-black border border-zinc-800 rounded-xl p-4 font-mono text-[11px] overflow-hidden flex flex-col">
          <div className="flex items-center gap-2 text-zinc-500 mb-3 pb-2 border-b border-zinc-900">
            <Terminal className="w-3 h-3" />
            <span className="uppercase tracking-wider">System Log</span>
          </div>
          <div className="flex-1 overflow-y-auto space-y-2 pr-2 text-zinc-400 scrollbar-thin">
            {logs.length === 0 ? (
              <span className="text-zinc-700 italic">
                System ready. Waiting for configuration...
              </span>
            ) : (
              logs.map((log, i) => (
                <div key={i} className="break-words">
                  <span className="opacity-50 mr-2">{">"}</span>
                  {log}
                </div>
              ))
            )}
            <div ref={logsEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdvancedGenerator;
