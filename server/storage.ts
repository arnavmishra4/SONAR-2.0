import { db } from "./db";
import {
  analyses,
  type Analysis,
  type InsertAnalysis,
} from "@shared/schema";
import { eq, desc } from "drizzle-orm";

export interface IStorage {
  getAnalyses(): Promise<Analysis[]>;
  getAnalysis(id: number): Promise<Analysis | undefined>;
  getDemoAnalysis(): Promise<Analysis | undefined>;
  createAnalysis(analysis: InsertAnalysis): Promise<Analysis>;
  updateAnalysisResults(id: number, results: any): Promise<Analysis>;
}

export class DatabaseStorage implements IStorage {
  async getAnalyses(): Promise<Analysis[]> {
    return await db.select().from(analyses).orderBy(desc(analyses.createdAt));
  }

  async getAnalysis(id: number): Promise<Analysis | undefined> {
    const [analysis] = await db.select().from(analyses).where(eq(analyses.id, id));
    return analysis;
  }

  async getDemoAnalysis(): Promise<Analysis | undefined> {
    const [analysis] = await db.select().from(analyses).where(eq(analyses.isDemo, true)).limit(1);
    return analysis;
  }

  async createAnalysis(insertAnalysis: InsertAnalysis): Promise<Analysis> {
    const [analysis] = await db
      .insert(analyses)
      .values(insertAnalysis)
      .returning();
    return analysis;
  }

  async updateAnalysisResults(id: number, results: any): Promise<Analysis> {
    const [updated] = await db
      .update(analyses)
      .set({ results, status: "completed" })
      .where(eq(analyses.id, id))
      .returning();
    return updated;
  }
}

export const storage = new DatabaseStorage();
