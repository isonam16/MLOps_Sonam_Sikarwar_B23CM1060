import React, { useEffect, useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { SystemMetric } from "../types";

interface SystemUsageChartProps {
  metrics?: Array<{
    timestamp: number;
    cpu: number;
    gpu: number;
    memory: number;
  }> | null;
}

const SystemUsageChart: React.FC<SystemUsageChartProps> = ({ metrics }) => {
  const [data, setData] = useState<SystemMetric[]>([]);

  useEffect(() => {
    // Generate initial data for live system monitoring
    const initialData = Array.from({ length: 20 }, (_, i) => ({
      time: new Date(Date.now() - (20 - i) * 1000).toLocaleTimeString([], {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      }),
      cpu: 20 + Math.random() * 10,
      memory: 40 + Math.random() * 5,
      gpu: 10 + Math.random() * 5,
    }));
    setData(initialData);

    const interval = setInterval(() => {
      setData((prevData) => {
        const now = new Date();
        const newPoint: SystemMetric = {
          time: now.toLocaleTimeString([], {
            hour12: false,
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
          }),
          cpu: Math.min(
            100,
            Math.max(
              0,
              prevData[prevData.length - 1].cpu + (Math.random() - 0.5) * 15
            )
          ),
          memory: Math.min(
            100,
            Math.max(
              0,
              prevData[prevData.length - 1].memory + (Math.random() - 0.5) * 5
            )
          ),
          gpu: Math.min(
            100,
            Math.max(
              0,
              prevData[prevData.length - 1].gpu + (Math.random() - 0.5) * 20
            )
          ),
        };
        return [...prevData.slice(1), newPoint];
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="w-full h-full bg-zinc-900/50 border border-zinc-800 rounded-xl p-4 flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
          System Resources
        </h3>
        <div className="flex gap-4 text-xs">
          <span className="flex items-center gap-1 text-emerald-400">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span> CPU
          </span>
          <span className="flex items-center gap-1 text-cyan-400">
            <span className="w-2 h-2 rounded-full bg-cyan-500"></span> GPU
          </span>
          <span className="flex items-center gap-1 text-purple-400">
            <span className="w-2 h-2 rounded-full bg-purple-500"></span> MEM
          </span>
        </div>
      </div>

      <div className="flex-1 w-full min-h-[150px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="colorCpu" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorGpu" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorMem" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#a855f7" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#a855f7" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="#27272a"
              vertical={false}
            />
            <XAxis
              dataKey="time"
              stroke="#52525b"
              tick={{ fill: "#52525b", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#52525b"
              tick={{ fill: "#52525b", fontSize: 10 }}
              tickLine={false}
              axisLine={false}
              domain={[0, 100]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                borderColor: "#27272a",
                color: "#e4e4e7",
              }}
              itemStyle={{ fontSize: "12px" }}
            />
            <Area
              type="monotone"
              dataKey="cpu"
              stroke="#10b981"
              fillOpacity={1}
              fill="url(#colorCpu)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="gpu"
              stroke="#06b6d4"
              fillOpacity={1}
              fill="url(#colorGpu)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="memory"
              stroke="#a855f7"
              fillOpacity={1}
              fill="url(#colorMem)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SystemUsageChart;
