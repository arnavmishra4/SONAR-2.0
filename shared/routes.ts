import { z } from 'zod';
import { insertAnalysisSchema, analyses } from './schema';
export type { CreateAnalysisRequest } from './schema';

export const errorSchemas = {
  validation: z.object({
    message: z.string(),
    field: z.string().optional(),
  }),
  notFound: z.object({
    message: z.string(),
  }),
  internal: z.object({
    message: z.string(),
  }),
};

// Demo data structure
export const demoDataSchema = z.object({
  title: z.string(),
  description: z.string(),
  summary: z.object({
    total_aois: z.number(),
    total_patches: z.number(),
    anomalies_detected: z.number(),
    anomaly_percentage: z.number(),
    mean_confidence: z.number(),
  }),
  aois: z.array(z.object({
    aoi_id: z.string(),
    total_patches: z.number(),
    anomalies_detected: z.number(),
    bounds: z.object({
      min_lat: z.number(),
      max_lat: z.number(),
      min_lon: z.number(),
      max_lon: z.number(),
    }),
    center: z.object({
      lat: z.number(),
      lng: z.number(),
    }),
  })),
  top_candidates: z.array(z.object({
    aoi_id: z.string(),
    patch_id: z.string(),
    confidence: z.number(),
    coordinates: z.object({
      lat: z.number(),
      lng: z.number(),
    }),
  })),
  all_anomalies: z.array(z.object({
    aoi_id: z.string(),
    patch_id: z.string(),
    confidence: z.number(),
    coordinates: z.object({
      lat: z.number(),
      lng: z.number(),
    }),
  })),
});

export const api = {
  analyses: {
    list: {
      method: 'GET' as const,
      path: '/api/analyses' as const,
      responses: {
        200: z.array(z.custom<typeof analyses.$inferSelect>()),
      },
    },
    get: {
      method: 'GET' as const,
      path: '/api/analyses/:id' as const,
      responses: {
        200: z.custom<typeof analyses.$inferSelect>(),
        404: errorSchemas.notFound,
      },
    },
    create: {
      method: 'POST' as const,
      path: '/api/analyses' as const,
      // In a real app, this would be multipart/form-data, but for the contract we validate metadata
      input: insertAnalysisSchema, 
      responses: {
        201: z.custom<typeof analyses.$inferSelect>(),
        400: errorSchemas.validation,
      },
    },
    // Special endpoint to get the "Demo" analysis directly
    getDemo: {
      method: 'GET' as const,
      path: '/api/demo' as const,
      responses: {
        200: z.custom<typeof analyses.$inferSelect>(),
      },
    },
    // NEW: Get comprehensive demo data with all AOIs and anomalies
    getDemoData: {
      method: 'GET' as const,
      path: '/api/demo-data' as const,
      responses: {
        200: demoDataSchema,
        404: errorSchemas.notFound,
      },
    }
  },
};

export function buildUrl(path: string, params?: Record<string, string | number>): string {
  let url = path;
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      if (url.includes(`:${key}`)) {
        url = url.replace(`:${key}`, String(value));
      }
    });
  }
  return url;
}