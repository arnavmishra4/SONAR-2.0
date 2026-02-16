import { useState } from "react";
import { ZoomIn, ZoomOut, Maximize, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeatmapViewerProps {
  imageUrl?: string;
  loading?: boolean;
}

export function HeatmapViewer({ imageUrl, loading }: HeatmapViewerProps) {
  const [zoom, setZoom] = useState(1);

  return (
    <div className="relative w-full aspect-square md:aspect-[4/3] bg-muted/30 rounded-xl overflow-hidden border border-border group">
      {/* Toolbar */}
      <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button 
          variant="secondary" 
          size="icon" 
          onClick={() => setZoom(z => Math.min(z + 0.5, 3))}
          className="shadow-lg bg-background/90 backdrop-blur"
        >
          <ZoomIn size={18} />
        </Button>
        <Button 
          variant="secondary" 
          size="icon" 
          onClick={() => setZoom(z => Math.max(z - 0.5, 1))}
          className="shadow-lg bg-background/90 backdrop-blur"
        >
          <ZoomOut size={18} />
        </Button>
        <Button 
          variant="secondary" 
          size="icon" 
          className="shadow-lg bg-background/90 backdrop-blur"
        >
          <Maximize size={18} />
        </Button>
      </div>

      {/* Map Content */}
      <div className="w-full h-full flex items-center justify-center overflow-hidden map-grid-bg relative">
        {loading ? (
          <div className="flex flex-col items-center gap-4 animate-pulse">
            <div className="w-16 h-16 border-4 border-primary/30 border-t-primary rounded-full animate-spin" />
            <p className="text-sm font-mono text-muted-foreground uppercase tracking-widest">Generating Terrain Analysis...</p>
          </div>
        ) : imageUrl ? (
          <div 
            className="w-full h-full transition-transform duration-300 ease-out"
            style={{ transform: `scale(${zoom})` }}
          >
            {/* HTML Comment describing the image content for Unsplash fallback */}
            {/* aerial view of archaeological dig site texture topographic map */}
            <img 
              src={imageUrl} 
              alt="Terrain Heatmap" 
              className="w-full h-full object-cover mix-blend-multiply opacity-90"
            />
            
            {/* Overlay Grid Effect */}
            <div className="absolute inset-0 pointer-events-none bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20" />
            
            {/* Simulated Heatmap Points */}
            <div className="absolute top-[30%] left-[40%] w-24 h-24 bg-red-500/30 rounded-full blur-xl animate-pulse" />
            <div className="absolute top-[60%] left-[70%] w-16 h-16 bg-yellow-500/30 rounded-full blur-lg" />
          </div>
        ) : (
          <div className="text-center text-muted-foreground p-8">
            <Layers className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No heatmap data available</p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-background/90 backdrop-blur px-3 py-2 rounded-lg border border-border shadow-sm text-xs font-mono">
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>High Probability (0.9+)</span>
        </div>
        <div className="flex items-center gap-2 mb-1">
          <div className="w-3 h-3 rounded-full bg-yellow-500" />
          <span>Medium Probability (0.7+)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-blue-500/50" />
          <span>Low Probability</span>
        </div>
      </div>
    </div>
  );
}
