import { createInsertSchema } from "drizzle-zod";
import { z } from "zod";

// === TABLE DEFINITIONS ===
export const analyses = pgTable("analyses", {
  id: serial("id").primaryKey(),
  title: text("title").notNull(),
  status: text("status").notNull().default("processing"), // processing, completed, failed
  isDemo: boolean("is_demo").default(false),
  
  // Storing results as JSONB to mimic the complex output structure from your Python script
  // In a real scenario, this might be a link to a file, but for the frontend demo, JSONB is perfect.
  results: jsonb("results"), 
  
  createdAt: timestamp("created_at").defaultNow(),
});

// === SCHEMAS ===
export const insertAnalysisSchema = createInsertSchema(analyses).omit({ 
  id: true, 
  createdAt: true, 
  status: true,
  results: true 
});

// === TYPES ===
export type Analysis = typeof analyses.$inferSelect;
export type InsertAnalysis = z.infer<typeof insertAnalysisSchema>;

// Frontend request types
export type CreateAnalysisRequest = InsertAnalysis;

// For the demo data structure (mimicking your python output)
export interface AnalysisResult {
  summary: {
    total_patches: number;
    anomalies_detected: number;
    anomaly_percentage: number;
    mean_confidence: number;
  };
  top_candidates: Array<{
    rank: number;
    patch_id: string;
    confidence: number;
    coordinates: { lat: number; lng: number };
  }>;
  heatmap_url?: string; // Base64 or URL
}