import { useState } from "react";
import { useLocation } from "wouter";
import { Navigation } from "@/components/Navigation";
import { UploadZone } from "@/components/UploadZone";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { useCreateAnalysis } from "@/hooks/use-analyses";
import { ArrowRight, Loader2 } from "lucide-react";
import { motion } from "framer-motion";

export default function Upload() {
  const [, setLocation] = useLocation();
  const { mutate: createAnalysis, isPending } = useCreateAnalysis();
  
  // In a real app, these would store actual File objects
  // For the demo, we just track if something is selected
  const [lidarFile, setLidarFile] = useState<File | null>(null);
  
  const handleAnalyze = () => {
    // We send a mock request since we don't have the actual Python backend connected for file processing
    // The backend will create a record with "processing" status
    createAnalysis(
      { 
        title: lidarFile ? `Analysis of ${lidarFile.name}` : "New Terrain Analysis",
        isDemo: true // Force demo mode for this prototype
      }, 
      {
        onSuccess: () => setLocation("/dashboard")
      }
    );
  };

  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />
      
      <main className="flex-1 container px-4 py-12 pt-24 max-w-5xl">
        <div className="mb-8">
          <h1 className="text-3xl font-display font-bold text-foreground">Start New Analysis</h1>
          <p className="text-muted-foreground mt-2">Upload your geospatial data to begin anomaly detection.</p>
        </div>

        <div className="grid lg:grid-cols-3 gap-8">
          <motion.div 
            className="lg:col-span-2 space-y-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <div className="grid sm:grid-cols-2 gap-4">
              <div className="sm:col-span-2">
                <UploadZone 
                  label="LiDAR DTM (Digital Terrain Model)" 
                  required 
                  onFileSelect={setLidarFile}
                />
              </div>
              <UploadZone label="Sentinel-2 Imagery" />
              <UploadZone label="HydroSHEDS Data" />
            </div>

            <Card className="bg-primary/5 border-primary/20">
              <CardContent className="p-6">
                <h3 className="font-bold text-primary mb-2 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
                  Processing Note
                </h3>
                <p className="text-sm text-muted-foreground">
                  The SONAR 2.0 ensemble model requires high-resolution inputs. 
                  Processing time varies based on terrain size (approx. 2-5 mins for 10km²).
                </p>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card className="sticky top-24">
              <CardHeader>
                <CardTitle>Analysis Settings</CardTitle>
                <CardDescription>Configure the detection parameters</CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Model Sensitivity</label>
                  <div className="h-2 bg-secondary rounded-full overflow-hidden">
                    <div className="h-full w-[70%] bg-primary" />
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Conservative</span>
                    <span>Aggressive</span>
                  </div>
                </div>

                <div className="space-y-4 pt-4 border-t border-border">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Files Selected</span>
                    <span className="font-mono font-bold">{lidarFile ? "1" : "0"}/3</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Est. Duration</span>
                    <span className="font-mono font-bold">~3 min</span>
                  </div>
                </div>

                <Button 
                  className="w-full h-12 text-lg shadow-lg hover:shadow-xl transition-all"
                  disabled={!lidarFile || isPending}
                  onClick={handleAnalyze}
                >
                  {isPending ? (
                    <>
                      <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                      Uploading...
                    </>
                  ) : (
                    <>
                      Begin Analysis
                      <ArrowRight className="ml-2 h-5 w-5" />
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
