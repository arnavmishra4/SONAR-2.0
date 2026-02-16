import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, buildUrl, type CreateAnalysisRequest } from "@shared/routes";
import { useToast } from "@/hooks/use-toast";

export function useAnalyses() {
  return useQuery({
    queryKey: [api.analyses.list.path],
    queryFn: async () => {
      const res = await fetch(api.analyses.list.path);
      if (!res.ok) throw new Error("Failed to fetch analyses");
      return api.analyses.list.responses[200].parse(await res.json());
    },
  });
}

export function useAnalysis(id: number) {
  return useQuery({
    queryKey: [api.analyses.get.path, id],
    queryFn: async () => {
      const url = buildUrl(api.analyses.get.path, { id });
      const res = await fetch(url);
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("Failed to fetch analysis");
      return api.analyses.get.responses[200].parse(await res.json());
    },
    enabled: !!id,
  });
}

export function useDemoAnalysis() {
  return useQuery({
    queryKey: [api.analyses.getDemo.path],
    queryFn: async () => {
      const res = await fetch(api.analyses.getDemo.path);
      if (!res.ok) throw new Error("Failed to fetch demo analysis");
      return api.analyses.getDemo.responses[200].parse(await res.json());
    },
  });
}

export function useCreateAnalysis() {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  
  return useMutation({
    mutationFn: async (data: CreateAnalysisRequest) => {
      const res = await fetch(api.analyses.create.path, {
        method: api.analyses.create.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      
      if (!res.ok) {
        if (res.status === 400) {
          const error = api.analyses.create.responses[400].parse(await res.json());
          throw new Error(error.message);
        }
        throw new Error("Failed to create analysis");
      }
      return api.analyses.create.responses[201].parse(await res.json());
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [api.analyses.list.path] });
      toast({
        title: "Analysis Started",
        description: "Your data is being processed by the SONAR 2.0 ensemble.",
      });
    },
    onError: (error) => {
      toast({
        variant: "destructive",
        title: "Error",
        description: error.message,
      });
    }
  });
}
