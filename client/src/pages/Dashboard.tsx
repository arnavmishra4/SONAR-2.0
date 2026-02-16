import { Navigation } from "@/components/Navigation";
import { DemoMap } from "@/components/DemoMap";

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />

      <main className="flex-1 container px-4 py-8 pt-24 max-w-7xl">
        {/* Header Section */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <h1 className="text-3xl font-display font-bold text-foreground">
              Demo Analysis - Multi-AOI Archaeological Site Detection
            </h1>
          </div>
          <p className="text-muted-foreground">
            Comprehensive analysis across 7 Areas of Interest showing detected archaeological anomalies using AI-powered detection
          </p>
        </div>

        {/* Demo Map Component */}
        <DemoMap />

        {/* Info Section */}
        <div className="mt-8 p-6 bg-muted/30 rounded-lg border border-border">
          <h3 className="font-semibold text-lg mb-4">About This Demo</h3>
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h4 className="font-semibold mb-2">🗺️ Geographic Coverage</h4>
              <p className="text-sm text-muted-foreground">
                This demo analyzes 7 distinct Areas of Interest (AOIs) from archaeological sites in the Amazon region. 
                Each AOI contains LiDAR-derived terrain data combined with satellite imagery.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">🤖 AI Detection Pipeline</h4>
              <p className="text-sm text-muted-foreground">
                SONAR 2.0 uses an ensemble of deep learning models including autoencoders, isolation forests, 
                K-means clustering, and a GATE meta-learner to identify archaeological anomalies with high precision.
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">📊 Analysis Metrics</h4>
              <p className="text-sm text-muted-foreground">
                Each detected anomaly is scored on confidence (0-100%). Red markers indicate high-confidence detections 
                (&gt;80%), orange for medium (60-80%), and yellow for lower confidence (&lt;60%).
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2">🔍 Interactive Exploration</h4>
              <p className="text-sm text-muted-foreground">
                Click on any red marker to view detailed information about the anomaly. Blue markers show AOI centers, 
                and blue rectangles indicate AOI boundaries. Zoom and pan to explore the full analysis.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}