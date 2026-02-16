import { useEffect, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Loader2, Layers, Mountain, Activity, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import Plot from 'react-plotly.js';

interface PatchDetailModalProps {
  isOpen: boolean;
  onClose: () => void;
  aoiId: string;
  patchId: string;
  confidence: number;
  coordinates: {
    lat: number;
    lng: number;
  };
}

interface ModelScores {
  autoencoder: number;
  isolation_forest: number;
  kmeans: number;
  similarity: number;
  gate_final: number;
}

export function PatchDetailModal({
  isOpen,
  onClose,
  aoiId,
  patchId,
  confidence,
  coordinates
}: PatchDetailModalProps) {
  const [loading, setLoading] = useState(false);
  const [visualization7Channel, setVisualization7Channel] = useState<string | null>(null);
  const [terrain3D, setTerrain3D] = useState<any>(null);
  const [modelScores, setModelScores] = useState<ModelScores | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Load all data when modal opens
  useEffect(() => {
    if (isOpen && aoiId && patchId) {
      loadPatchData();
    }
  }, [isOpen, aoiId, patchId]);

  const loadPatchData = async () => {
    setLoading(true);
    setError(null);

    try {
      // Load 7-channel visualization
      const viz7Response = await fetch(`/api/patch/${aoiId}/${patchId}/visualization`);
      if (viz7Response.ok) {
        const viz7Data = await viz7Response.json();
        if (viz7Data.success) {
          setVisualization7Channel(viz7Data.image);
        }
      }

      // Load 3D terrain
      const terrain3DResponse = await fetch(`/api/patch/${aoiId}/${patchId}/terrain3d`);
      if (terrain3DResponse.ok) {
        const terrain3DData = await terrain3DResponse.json();
        if (terrain3DData.success) {
          setTerrain3D(terrain3DData.plotly_json);
        }
      }

      // Load model scores
      const scoresResponse = await fetch(`/api/patch/${aoiId}/${patchId}/scores`);
      if (scoresResponse.ok) {
        const scoresData = await scoresResponse.json();
        if (scoresData.success) {
          setModelScores(scoresData.scores);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load patch data');
    } finally {
      setLoading(false);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.9) return 'text-red-600';
    if (score >= 0.7) return 'text-orange-500';
    if (score >= 0.5) return 'text-yellow-500';
    return 'text-green-600';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 0.9) return '🔴';
    if (score >= 0.7) return '🟠';
    if (score >= 0.5) return '🟡';
    return '🟢';
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto z-[9999]">
        <DialogHeader>
          <div className="flex items-center justify-between">
            <DialogTitle className="text-2xl">
              🏛️ Patch Analysis - {patchId}
            </DialogTitle>
            <Button variant="ghost" size="icon" onClick={onClose}>
              <X className="h-4 w-4" />
            </Button>
          </div>
          <div className="flex gap-2 mt-2">
            <Badge variant="outline">{aoiId}</Badge>
            <Badge variant={confidence >= 0.9 ? 'destructive' : 'secondary'}>
              Confidence: {(confidence * 100).toFixed(1)}%
            </Badge>
            <Badge variant="outline" className="font-mono text-xs">
              {coordinates.lat.toFixed(6)}, {coordinates.lng.toFixed(6)}
            </Badge>
          </div>
        </DialogHeader>

        {loading && (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <span className="ml-3 text-muted-foreground">Loading patch data...</span>
          </div>
        )}

        {error && (
          <div className="p-4 bg-destructive/10 text-destructive rounded-lg">
            ⚠️ {error}
          </div>
        )}

        {!loading && !error && (
          <Tabs defaultValue="7channel" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="7channel" className="flex items-center gap-2">
                <Layers className="w-4 h-4" />
                7-Channel View
              </TabsTrigger>
              <TabsTrigger value="3d" className="flex items-center gap-2">
                <Mountain className="w-4 h-4" />
                3D Terrain
              </TabsTrigger>
              <TabsTrigger value="scores" className="flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Model Scores
              </TabsTrigger>
            </TabsList>

            {/* 7-Channel Visualization */}
            <TabsContent value="7channel" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Multi-Channel Analysis</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Seven input channels used by the AI models: DTM, Slope, Roughness, NDVI, NDWI, Flow Accumulation, Flow Direction
                  </p>
                </CardHeader>
                <CardContent>
                  {visualization7Channel ? (
                    <img 
                      src={visualization7Channel} 
                      alt="7-Channel Visualization" 
                      className="w-full rounded-lg border"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-64 bg-muted rounded-lg">
                      <p className="text-muted-foreground">7-channel visualization not available</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* 3D Terrain */}
            <TabsContent value="3d" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">3D Terrain Model</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Interactive 3D visualization of the LiDAR-derived Digital Terrain Model
                  </p>
                </CardHeader>
                <CardContent>
                  {terrain3D ? (
                    <div className="w-full h-[500px]">
                      <Plot
                        data={terrain3D.data}
                        layout={terrain3D.layout}
                        config={{ responsive: true }}
                        style={{ width: '100%', height: '100%' }}
                      />
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-64 bg-muted rounded-lg">
                      <p className="text-muted-foreground">3D terrain not available</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            {/* Model Scores */}
            <TabsContent value="scores" className="mt-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">AI Model Scores</CardTitle>
                  <p className="text-sm text-muted-foreground">
                    Individual model predictions and final GATE ensemble score
                  </p>
                </CardHeader>
                <CardContent>
                  {modelScores ? (
                    <div className="space-y-4">
                      {/* Individual Model Scores */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {/* Autoencoder */}
                        <div className="p-4 bg-muted/50 rounded-lg border-l-4 border-blue-500">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="font-semibold">Autoencoder</h4>
                              <p className="text-xs text-muted-foreground">Reconstruction error</p>
                            </div>
                            <div className="text-right">
                              <span className="text-2xl">{getScoreIcon(modelScores.autoencoder)}</span>
                              <p className={`text-xl font-bold ${getScoreColor(modelScores.autoencoder)}`}>
                                {(modelScores.autoencoder * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* K-Means */}
                        <div className="p-4 bg-muted/50 rounded-lg border-l-4 border-purple-500">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="font-semibold">K-Means</h4>
                              <p className="text-xs text-muted-foreground">Cluster distance</p>
                            </div>
                            <div className="text-right">
                              <span className="text-2xl">{getScoreIcon(modelScores.kmeans)}</span>
                              <p className={`text-xl font-bold ${getScoreColor(modelScores.kmeans)}`}>
                                {(modelScores.kmeans * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Similarity */}
                        <div className="p-4 bg-muted/50 rounded-lg border-l-4 border-cyan-500">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="font-semibold">Similarity</h4>
                              <p className="text-xs text-muted-foreground">Reference comparison</p>
                            </div>
                            <div className="text-right">
                              <span className="text-2xl">{getScoreIcon(modelScores.similarity)}</span>
                              <p className={`text-xl font-bold ${getScoreColor(modelScores.similarity)}`}>
                                {(modelScores.similarity * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Isolation Forest */}
                        <div className="p-4 bg-muted/50 rounded-lg border-l-4 border-green-500">
                          <div className="flex items-center justify-between">
                            <div>
                              <h4 className="font-semibold">Isolation Forest</h4>
                              <p className="text-xs text-muted-foreground">Anomaly isolation</p>
                            </div>
                            <div className="text-right">
                              <span className="text-2xl">{getScoreIcon(modelScores.isolation_forest)}</span>
                              <p className={`text-xl font-bold ${getScoreColor(modelScores.isolation_forest)}`}>
                                {(modelScores.isolation_forest * 100).toFixed(1)}%
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* GATE Final Score */}
                      <div className="p-6 bg-gradient-to-br from-purple-500/10 to-blue-500/10 rounded-lg border-2 border-purple-500/50">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-xl font-bold">GATE Final Score</h3>
                            <p className="text-sm text-muted-foreground">Ensemble meta-learner prediction</p>
                          </div>
                          <div className="text-right">
                            <span className="text-5xl">{getScoreIcon(modelScores.gate_final)}</span>
                            <p className={`text-4xl font-bold ${getScoreColor(modelScores.gate_final)}`}>
                              {(modelScores.gate_final * 100).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Legend */}
                      <div className="p-4 bg-muted/30 rounded-lg border text-sm">
                        <p className="font-semibold mb-2">Score Interpretation:</p>
                        <div className="grid grid-cols-2 gap-2">
                          <div className="flex items-center gap-2">
                            <span>🟢</span>
                            <span>Low (0-50%): Normal terrain</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span>🟡</span>
                            <span>Medium (50-70%): Possible feature</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span>🟠</span>
                            <span>High (70-90%): Likely anomaly</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span>🔴</span>
                            <span>Critical (90-100%): Strong detection</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-64 bg-muted rounded-lg">
                      <p className="text-muted-foreground">Model scores not available</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </DialogContent>
    </Dialog>
  );
}