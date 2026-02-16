import { useEffect, useState, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { Loader2, MapPin, AlertCircle, Filter, Layers as LayersIcon } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { PatchDetailModal } from "@/components/PatchDetailModal";

// Fix Leaflet default marker icon issue
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface DemoMapProps {
  className?: string;
}

interface Anomaly {
  aoi_id: string;
  patch_id: string;
  confidence: number;
  coordinates: {
    lat: number;
    lng: number;
  };
  pixel_location?: {
    row: number;
    col: number;
  };
}

interface AOIInfo {
  aoi_id: string;
  total_patches: number;
  anomalies_detected: number;
  bounds: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
  center: {
    lat: number;
    lng: number;
  };
}

interface DemoData {
  title: string;
  description: string;
  summary: {
    total_aois: number;
    total_patches: number;
    anomalies_detected: number;
    anomaly_percentage: number;
    mean_confidence: number;
  };
  aois: AOIInfo[];
  top_candidates: Anomaly[];
  all_anomalies: Anomaly[];
}

export function DemoMap({ className }: DemoMapProps) {
  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const markersLayerRef = useRef<L.LayerGroup | null>(null);
  const terrainOverlaysRef = useRef<{[key: string]: L.ImageOverlay}>({});
  
  const [demoData, setDemoData] = useState<DemoData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnomaly, setSelectedAnomaly] = useState<Anomaly | null>(null);
  
  // Modal state
  const [modalOpen, setModalOpen] = useState(false);
  const [modalData, setModalData] = useState<{
    aoiId: string;
    patchId: string;
    confidence: number;
    coordinates: { lat: number; lng: number };
  } | null>(null);
  
  // Filter controls
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.9); // Default 90%
  const [terrainLayerType, setTerrainLayerType] = useState<'relief' | 'hillshade' | 'basic'>('relief');
  const [showTerrainOverlay, setShowTerrainOverlay] = useState(true);

  // Load demo data
  useEffect(() => {
    const loadDemoData = async () => {
      try {
        const response = await fetch('/api/demo-data');
        if (!response.ok) {
          throw new Error('Failed to load demo data');
        }
        const data = await response.json();
        setDemoData(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    loadDemoData();
  }, []);

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || !demoData || mapRef.current) return;

    const allLats = demoData.aois.flatMap(aoi => [aoi.bounds.min_lat, aoi.bounds.max_lat]);
    const allLngs = demoData.aois.flatMap(aoi => [aoi.bounds.min_lon, aoi.bounds.max_lon]);
    
    const centerLat = (Math.min(...allLats) + Math.max(...allLats)) / 2;
    const centerLng = (Math.min(...allLngs) + Math.max(...allLngs)) / 2;

    const map = L.map(mapContainerRef.current).setView([centerLat, centerLng], 12);

    // Base layers
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
      attribution: 'Tiles © Esri',
      maxZoom: 19,
    });

    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors',
      maxZoom: 19,
    });

    const terrainLayer = L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png', {
      attribution: 'Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap',
      maxZoom: 17,
    });

    satelliteLayer.addTo(map);

    const baseMaps = {
      "🛰️ Satellite": satelliteLayer,
      "🗺️ OpenStreetMap": osmLayer,
      "🏔️ Terrain": terrainLayer,
    };

    // Add AOI boundaries
    const aoiBoundsLayer = L.layerGroup();
    demoData.aois.forEach(aoi => {
      const bounds: L.LatLngBoundsLiteral = [
        [aoi.bounds.min_lat, aoi.bounds.min_lon],
        [aoi.bounds.max_lat, aoi.bounds.max_lon]
      ];

      L.rectangle(bounds, {
        color: '#3b82f6',
        weight: 2,
        fillOpacity: 0.05,
      }).addTo(aoiBoundsLayer).bindPopup(`
        <div class="font-sans p-2">
          <h3 class="font-bold text-sm mb-2">${aoi.aoi_id}</h3>
          <p class="text-xs">📦 Patches: ${aoi.total_patches}</p>
          <p class="text-xs">🔴 Anomalies: ${aoi.anomalies_detected}</p>
          <p class="text-xs">📊 Rate: ${((aoi.anomalies_detected / aoi.total_patches) * 100).toFixed(1)}%</p>
        </div>
      `);

      L.marker([aoi.center.lat, aoi.center.lng], {
        icon: L.divIcon({
          className: 'custom-div-icon',
          html: `<div class="bg-blue-500 text-white rounded-full w-8 h-8 flex items-center justify-center text-xs font-bold shadow-lg border-2 border-white">${aoi.aoi_id.split('_')[1]}</div>`,
          iconSize: [32, 32],
          iconAnchor: [16, 16]
        })
      }).addTo(aoiBoundsLayer).bindPopup(`
        <div class="font-sans p-2">
          <h3 class="font-bold mb-2">${aoi.aoi_id} Center</h3>
          <p class="text-xs font-mono">Lat: ${aoi.center.lat.toFixed(6)}</p>
          <p class="text-xs font-mono">Lng: ${aoi.center.lng.toFixed(6)}</p>
        </div>
      `);
    });

    aoiBoundsLayer.addTo(map);

    const markersLayer = L.layerGroup();
    markersLayer.addTo(map);
    markersLayerRef.current = markersLayer;

    const overlayMaps = {
      "📍 AOI Boundaries": aoiBoundsLayer,
      "🔴 Anomaly Markers": markersLayer,
    };

    L.control.layers(baseMaps, overlayMaps, { position: 'topright' }).addTo(map);

    const allBounds = demoData.aois.map(aoi => [
      [aoi.bounds.min_lat, aoi.bounds.min_lon],
      [aoi.bounds.max_lat, aoi.bounds.max_lon]
    ]).flat() as L.LatLngTuple[];
    
    if (allBounds.length > 0) {
      map.fitBounds(allBounds as L.LatLngBoundsLiteral, { padding: [50, 50] });
    }

    L.control.scale({ metric: true, imperial: false, position: 'bottomleft' }).addTo(map);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [demoData]);

  // Load terrain overlays
  useEffect(() => {
    if (!mapRef.current || !demoData || !showTerrainOverlay) {
      // Remove existing overlays if terrain is disabled
      Object.values(terrainOverlaysRef.current).forEach(overlay => {
        if (mapRef.current) {
          mapRef.current.removeLayer(overlay);
        }
      });
      terrainOverlaysRef.current = {};
      return;
    }

    const loadTerrainOverlay = async (aoi: AOIInfo) => {
      try {
        const response = await fetch(`/api/terrain/${aoi.aoi_id}?layer_type=${terrainLayerType}`);
        if (!response.ok) return;
        
        const data = await response.json();
        
        if (data.success && mapRef.current) {
          const bounds: L.LatLngBoundsLiteral = [
            [data.bounds.min_lat, data.bounds.min_lon],
            [data.bounds.max_lat, data.bounds.max_lon]
          ];
          
          // Remove existing overlay
          if (terrainOverlaysRef.current[aoi.aoi_id]) {
            mapRef.current.removeLayer(terrainOverlaysRef.current[aoi.aoi_id]);
          }
          
          // Add new overlay
          const overlay = L.imageOverlay(
            data.image,
            bounds,
            {
              opacity: 0.75,
              interactive: false
            }
          ).addTo(mapRef.current);
          
          terrainOverlaysRef.current[aoi.aoi_id] = overlay;
        }
      } catch (err) {
        console.error(`Failed to load terrain for ${aoi.aoi_id}:`, err);
      }
    };

    // Load for all AOIs
    demoData.aois.forEach(aoi => loadTerrainOverlay(aoi));

  }, [demoData, terrainLayerType, showTerrainOverlay]);

  // Update markers when threshold changes
  useEffect(() => {
    if (!mapRef.current || !demoData || !markersLayerRef.current) return;

    const markersLayer = markersLayerRef.current;
    markersLayer.clearLayers();

    const filteredAnomalies = demoData.all_anomalies.filter(
      anomaly => anomaly.confidence >= confidenceThreshold
    );

    const anomalyIcon = (confidence: number) => {
      let color: string;
      let size: number;
      
      if (confidence >= 0.95) {
        color = '#dc2626';
        size = 14;
      } else if (confidence >= 0.9) {
        color = '#ef4444';
        size = 12;
      } else if (confidence >= 0.8) {
        color = '#f97316';
        size = 10;
      } else {
        color = '#eab308';
        size = 8;
      }
      
      return L.divIcon({
        className: 'custom-div-icon',
        html: `<div class="rounded-full shadow-lg animate-pulse" style="background-color: ${color}; width: ${size}px; height: ${size}px; border: 2px solid white;"></div>`,
        iconSize: [size, size],
        iconAnchor: [size / 2, size / 2]
      });
    };

    filteredAnomalies.forEach((anomaly) => {
      const marker = L.marker(
        [anomaly.coordinates.lat, anomaly.coordinates.lng],
        { icon: anomalyIcon(anomaly.confidence) }
      ).addTo(markersLayer);

      const confidencePercent = (anomaly.confidence * 100).toFixed(1);
      const confidenceColor = anomaly.confidence >= 0.95 ? '#dc2626' : 
                              anomaly.confidence >= 0.9 ? '#ef4444' : 
                              anomaly.confidence >= 0.8 ? '#f97316' : '#eab308';

      marker.bindPopup(`
        <div class="font-sans p-3 min-w-[200px]">
          <div class="flex items-center gap-2 mb-2">
            <div class="w-3 h-3 rounded-full" style="background-color: ${confidenceColor}"></div>
            <h3 class="font-bold text-sm">🏛️ Archaeological Anomaly</h3>
          </div>
          <hr class="my-2 border-gray-200" />
          <div class="space-y-1 text-xs">
            <p><span class="font-semibold">AOI:</span> ${anomaly.aoi_id}</p>
            <p><span class="font-semibold">Patch:</span> ${anomaly.patch_id}</p>
            <p><span class="font-semibold">Confidence:</span> 
              <span class="font-bold" style="color: ${confidenceColor}">${confidencePercent}%</span>
            </p>
            <p class="font-mono text-[10px] text-gray-600">
              ${anomaly.coordinates.lat.toFixed(6)}, ${anomaly.coordinates.lng.toFixed(6)}
            </p>
          </div>
          <button 
            onclick="window.openPatchDetail('${anomaly.aoi_id}', '${anomaly.patch_id}', ${anomaly.confidence}, ${anomaly.coordinates.lat}, ${anomaly.coordinates.lng})"
            class="mt-3 w-full px-3 py-1.5 bg-blue-500 text-white rounded text-xs font-semibold hover:bg-blue-600"
          >
            🔬 View Detailed Analysis
          </button>
        </div>
      `);

      marker.on('click', () => {
        setSelectedAnomaly(anomaly);
      });
    });

  }, [demoData, confidenceThreshold]);

  // Make openPatchDetail available globally for popup button
  useEffect(() => {
    (window as any).openPatchDetail = (aoiId: string, patchId: string, confidence: number, lat: number, lng: number) => {
      setModalData({
        aoiId,
        patchId,
        confidence,
        coordinates: { lat, lng }
      });
      setModalOpen(true);
    };

    return () => {
      delete (window as any).openPatchDetail;
    };
  }, []);

  if (loading) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="flex flex-col items-center gap-4">
            <Loader2 className="w-12 h-12 animate-spin text-primary" />
            <p className="text-muted-foreground">Loading demo data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className={className}>
        <CardContent className="flex items-center justify-center h-96">
          <div className="flex flex-col items-center gap-4 text-center">
            <AlertCircle className="w-12 h-12 text-destructive" />
            <div>
              <p className="font-semibold text-destructive">Failed to load demo data</p>
              <p className="text-sm text-muted-foreground mt-2">{error}</p>
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const filteredCount = demoData?.all_anomalies.filter(a => a.confidence >= confidenceThreshold).length || 0;

  return (
    <div className={className}>
      {/* Statistics Header */}
      {demoData && (
        <Card className="mb-4">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <MapPin className="w-5 h-5 text-primary" />
              {demoData.title}
            </CardTitle>
            <p className="text-sm text-muted-foreground">{demoData.description}</p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <p className="text-2xl font-bold text-primary">{demoData.summary.total_aois}</p>
                <p className="text-xs text-muted-foreground">AOIs</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold">{demoData.summary.total_patches.toLocaleString()}</p>
                <p className="text-xs text-muted-foreground">Patches</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-red-500">{demoData.summary.anomalies_detected}</p>
                <p className="text-xs text-muted-foreground">Total Anomalies</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-orange-500">{demoData.summary.anomaly_percentage.toFixed(2)}%</p>
                <p className="text-xs text-muted-foreground">Detection Rate</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-blue-500">{(demoData.summary.mean_confidence * 100).toFixed(1)}%</p>
                <p className="text-xs text-muted-foreground">Avg Confidence</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Filter Controls */}
      <Card className="mb-4">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Filter className="w-4 h-4" />
            Detection & Visualization Controls
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Confidence Threshold */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="confidence-slider" className="text-sm font-medium">
                Confidence Threshold: {(confidenceThreshold * 100).toFixed(0)}%
              </Label>
              <Badge variant={confidenceThreshold >= 0.9 ? "destructive" : "secondary"}>
                {filteredCount} anomalies shown
              </Badge>
            </div>
            <Slider
              id="confidence-slider"
              min={0.5}
              max={1.0}
              step={0.05}
              value={[confidenceThreshold]}
              onValueChange={(values) => setConfidenceThreshold(values[0])}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>50% (All)</span>
              <span>90% (High Confidence)</span>
              <span>100% (Critical)</span>
            </div>
            <div className="flex gap-2 flex-wrap mt-2">
              <Button 
                variant={confidenceThreshold === 0.5 ? "default" : "outline"}
                size="sm"
                onClick={() => setConfidenceThreshold(0.5)}
              >
                All (50%+)
              </Button>
              <Button 
                variant={confidenceThreshold === 0.8 ? "default" : "outline"}
                size="sm"
                onClick={() => setConfidenceThreshold(0.8)}
              >
                Medium (80%+)
              </Button>
              <Button 
                variant={confidenceThreshold === 0.9 ? "default" : "outline"}
                size="sm"
                onClick={() => setConfidenceThreshold(0.9)}
              >
                High (90%+)
              </Button>
              <Button 
                variant={confidenceThreshold === 0.95 ? "default" : "outline"}
                size="sm"
                onClick={() => setConfidenceThreshold(0.95)}
              >
                Critical (95%+)
              </Button>
            </div>
          </div>

          {/* Terrain Overlay Controls */}
          <div className="space-y-3 pt-4 border-t">
            <Label className="text-sm font-medium flex items-center gap-2">
              <LayersIcon className="w-4 h-4" />
              LiDAR Terrain Overlay
            </Label>
            <div className="flex gap-2 flex-wrap">
              <Button
                variant={showTerrainOverlay && terrainLayerType === 'relief' ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setTerrainLayerType('relief');
                  setShowTerrainOverlay(true);
                }}
              >
                🏛️ Archaeological Relief
              </Button>
              <Button
                variant={showTerrainOverlay && terrainLayerType === 'hillshade' ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setTerrainLayerType('hillshade');
                  setShowTerrainOverlay(true);
                }}
              >
                🗻 Hillshade
              </Button>
              <Button
                variant={showTerrainOverlay && terrainLayerType === 'basic' ? "default" : "outline"}
                size="sm"
                onClick={() => {
                  setTerrainLayerType('basic');
                  setShowTerrainOverlay(true);
                }}
              >
                🌍 Basic Terrain
              </Button>
              <Button
                variant={!showTerrainOverlay ? "default" : "outline"}
                size="sm"
                onClick={() => setShowTerrainOverlay(false)}
              >
                ❌ Hide Terrain
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Custom LiDAR-derived terrain overlays show micro-topographic features ideal for archaeological detection
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Map */}
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between flex-wrap gap-4">
            <CardTitle className="flex items-center gap-2">
              <LayersIcon className="w-5 h-5" />
              Interactive Archaeological Map
            </CardTitle>
            <div className="flex gap-2 flex-wrap">
              <Badge variant="default" className="bg-red-600">🔴 Critical (95%+)</Badge>
              <Badge variant="destructive">🔴 High (90%+)</Badge>
              <Badge variant="outline" className="bg-orange-500 text-white">🟠 Medium (80%+)</Badge>
              <Badge variant="secondary">🟡 Low (50%+)</Badge>
            </div>
          </div>
          <p className="text-sm text-muted-foreground mt-2">
            🗺️ Switch between Satellite, Terrain views • 
            🎯 Adjust filters above • 
            🔴 Click markers for detailed analysis
          </p>
        </CardHeader>
        <CardContent className="p-0">
          <div 
            ref={mapContainerRef} 
            className="w-full h-[700px] rounded-b-lg"
          />
        </CardContent>
      </Card>

      {/* Selected Anomaly Details */}
      {selectedAnomaly && (
        <Card className="mt-4 border-2 border-primary">
          <CardHeader className="bg-primary/5">
            <CardTitle className="text-lg flex items-center justify-between">
              <span>🎯 Selected Anomaly</span>
              <Button
                onClick={() => {
                  setModalData({
                    aoiId: selectedAnomaly.aoi_id,
                    patchId: selectedAnomaly.patch_id,
                    confidence: selectedAnomaly.confidence,
                    coordinates: selectedAnomaly.coordinates
                  });
                  setModalOpen(true);
                }}
              >
                🔬 View Full Analysis
              </Button>
            </CardTitle>
          </CardHeader>
          <CardContent className="pt-6">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-6">
              <div>
                <p className="text-sm font-semibold text-muted-foreground">AOI ID</p>
                <p className="text-lg font-bold">{selectedAnomaly.aoi_id}</p>
              </div>
              <div>
                <p className="text-sm font-semibold text-muted-foreground">Patch ID</p>
                <p className="text-lg font-mono">{selectedAnomaly.patch_id}</p>
              </div>
              <div>
                <p className="text-sm font-semibold text-muted-foreground">Confidence Score</p>
                <Badge 
                  variant={selectedAnomaly.confidence >= 0.95 ? 'default' : selectedAnomaly.confidence >= 0.9 ? 'destructive' : 'secondary'}
                  className="text-base px-3 py-1"
                >
                  {(selectedAnomaly.confidence * 100).toFixed(2)}%
                </Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Patch Detail Modal */}
      {modalData && (
        <PatchDetailModal
          isOpen={modalOpen}
          onClose={() => setModalOpen(false)}
          aoiId={modalData.aoiId}
          patchId={modalData.patchId}
          confidence={modalData.confidence}
          coordinates={modalData.coordinates}
        />
      )}
    </div>
  );
}