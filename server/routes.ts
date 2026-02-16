import type { Express } from "express";
import type { Server } from "http";
import { storage } from "./storage";
import { api } from "@shared/routes";
import { z } from "zod";
import multer from "multer";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const upload = multer({ storage: multer.memoryStorage() });

// For ES modules, get the directory name
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  
  // Seed the demo data if it doesn't exist
  await seedDemoData();

  app.get(api.analyses.list.path, async (req, res) => {
    const analyses = await storage.getAnalyses();
    res.json(analyses);
  });

  app.get(api.analyses.get.path, async (req, res) => {
    const analysis = await storage.getAnalysis(Number(req.params.id));
    if (!analysis) {
      return res.status(404).json({ message: 'Analysis not found' });
    }
    res.json(analysis);
  });

  app.get(api.analyses.getDemo.path, async (req, res) => {
    const demo = await storage.getDemoAnalysis();
    if (!demo) {
      // Fallback if seed didn't run for some reason
      return res.status(404).json({ message: 'Demo not initialized' });
    }
    res.json(demo);
  });

  // NEW: Get comprehensive demo data with all AOIs and anomalies
  app.get(api.analyses.getDemoData.path, async (req, res) => {
    try {
      // Try multiple possible locations for demo_data.json
      const possiblePaths = [
        // In the server directory
        path.join(__dirname, 'demo_data.json'),
        // In the project root
        path.join(process.cwd(), 'demo_data.json'),
        // One level up from server directory
        path.join(__dirname, '..', 'demo_data.json'),
        // In a data directory
        path.join(process.cwd(), 'data', 'demo_data.json'),
        // In the server's parent directory
        path.join(__dirname, '..', '..', 'demo_data.json'),
      ];
      
      let demoDataPath: string | null = null;
      
      // Find the first path that exists
      for (const testPath of possiblePaths) {
        if (fs.existsSync(testPath)) {
          demoDataPath = testPath;
          console.log(`✅ Found demo_data.json at: ${testPath}`);
          break;
        } else {
          console.log(`❌ Not found at: ${testPath}`);
        }
      }
      
      if (!demoDataPath) {
        console.error('Demo data file not found in any of the expected locations');
        return res.status(404).json({ 
          message: 'Demo data not found. Please ensure demo_data.json is in the project root or server directory.',
          searchedPaths: possiblePaths
        });
      }
      
      const demoData = JSON.parse(fs.readFileSync(demoDataPath, 'utf-8'));
      res.json(demoData);
      
    } catch (error) {
      console.error('Error loading demo data:', error);
      res.status(500).json({ 
        message: 'Failed to load demo data',
        error: error instanceof Error ? error.message : 'Unknown error'
      });
    }
  });

  // Handle file uploads + metadata creation
  app.post(api.analyses.create.path, upload.array('files'), async (req, res) => {
    try {
      // In a real app, we'd process the files here.
      // For now, we just create the record and simulate processing.
      
      // Parse the body manually since multer handles the files
      const title = req.body.title || "New Analysis";
      
      const analysis = await storage.createAnalysis({
        title,
        isDemo: false,
      });

      // Simulate async processing (Python script)
      setTimeout(async () => {
        await storage.updateAnalysisResults(analysis.id, MOCK_RESULTS);
      }, 3000);

      res.status(201).json(analysis);
    } catch (err) {
      console.error(err);
      res.status(500).json({ message: "Internal server error" });
    }
  });

  return httpServer;
}

// Mock Data derived from your Python script structure
const MOCK_RESULTS = {
  summary: {
    total_patches: 1024,
    anomalies_detected: 42,
    anomaly_percentage: 4.1,
    mean_confidence: 0.76,
  },
  top_candidates: [
    { rank: 1, patch_id: "P_1024_A", confidence: 0.98, coordinates: { lat: 51.1789, lng: -1.8262 } },
    { rank: 2, patch_id: "P_0056_B", confidence: 0.95, coordinates: { lat: 51.1795, lng: -1.8255 } },
    { rank: 3, patch_id: "P_0891_C", confidence: 0.92, coordinates: { lat: 51.1772, lng: -1.8248 } },
    { rank: 4, patch_id: "P_0234_D", confidence: 0.89, coordinates: { lat: 51.1765, lng: -1.8270 } },
    { rank: 5, patch_id: "P_0112_E", confidence: 0.85, coordinates: { lat: 51.1780, lng: -1.8290 } },
  ],
  // This is a placeholder; frontend will likely use a real image or a generated canvas
  heatmap_url: "placeholder" 
};

async function seedDemoData() {
  const existing = await storage.getDemoAnalysis();
  if (!existing) {
    await storage.createAnalysis({
      title: "Demo: Multi-AOI Archaeological Analysis",
      isDemo: true,
    });
    // Immediately complete the demo
    const demo = await storage.getDemoAnalysis();
    if (demo) {
      await storage.updateAnalysisResults(demo.id, MOCK_RESULTS);
    }
  }
}