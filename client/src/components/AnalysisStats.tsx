import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { AlertCircle, Target, Brain, Activity } from "lucide-react";
import type { AnalysisResult } from "@shared/schema";

interface AnalysisStatsProps {
  data: AnalysisResult;
}

export function AnalysisStats({ data }: AnalysisStatsProps) {
  // Mock data for the consensus chart if not present in summary
  const consensusData = [
    { name: "Autoencoder", score: 85, fill: "#A0522D" }, // Sienna
    { name: "iForest", score: 72, fill: "#CD853F" },    // Peru
    { name: "K-Means", score: 65, fill: "#DAA520" },    // Goldenrod
    { name: "GATE", score: 92, fill: "#556B2F" },       // Dark Olive Green
  ];

  return (
    <div className="space-y-6">
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
          <CardContent className="p-4 flex flex-col items-start gap-2">
            <span className="text-muted-foreground text-xs font-mono uppercase tracking-widest flex items-center gap-1">
              <Activity className="w-3 h-3 text-accent" /> Total Patches
            </span>
            <span className="text-2xl font-mono font-bold text-foreground">
              {data.summary.total_patches.toLocaleString()}
            </span>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
          <CardContent className="p-4 flex flex-col items-start gap-2">
            <span className="text-muted-foreground text-xs font-mono uppercase tracking-widest flex items-center gap-1">
              <Target className="w-3 h-3 text-destructive" /> Anomalies
            </span>
            <span className="text-2xl font-mono font-bold text-foreground">
              {data.summary.anomalies_detected}
            </span>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
          <CardContent className="p-4 flex flex-col items-start gap-2">
            <span className="text-muted-foreground text-xs font-mono uppercase tracking-widest flex items-center gap-1">
              <AlertCircle className="w-3 h-3 text-accent" /> Detection %
            </span>
            <span className="text-2xl font-mono font-bold text-foreground">
              {data.summary.anomaly_percentage}%
            </span>
          </CardContent>
        </Card>

        <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
          <CardContent className="p-4 flex flex-col items-start gap-2">
            <span className="text-muted-foreground text-xs font-mono uppercase tracking-widest flex items-center gap-1">
              <Brain className="w-3 h-3 text-primary" /> Mean Conf.
            </span>
            <span className="text-2xl font-mono font-bold text-foreground">
              {(data.summary.mean_confidence * 100).toFixed(1)}%
            </span>
          </CardContent>
        </Card>
      </div>

      {/* Model Consensus Chart */}
      <Card className="border-none shadow-none bg-transparent">
        <CardHeader className="px-0">
          <CardTitle className="text-lg font-display">Model Consensus</CardTitle>
        </CardHeader>
        <CardContent className="p-0 h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={consensusData} layout="vertical" margin={{ left: 0 }}>
              <XAxis type="number" hide />
              <YAxis 
                dataKey="name" 
                type="category" 
                axisLine={false} 
                tickLine={false}
                tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12, fontFamily: 'var(--font-mono)' }}
                width={100}
              />
              <Tooltip 
                cursor={{ fill: 'transparent' }}
                contentStyle={{ 
                  backgroundColor: 'hsl(var(--card))', 
                  borderColor: 'hsl(var(--border))',
                  borderRadius: 'var(--radius)',
                  fontFamily: 'var(--font-mono)'
                }}
              />
              <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={24}>
                {consensusData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
