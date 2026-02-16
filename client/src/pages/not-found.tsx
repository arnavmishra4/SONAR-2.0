import { Link } from "wouter";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function NotFound() {
  return (
    <div className="min-h-screen w-full flex items-center justify-center bg-background">
      <div className="text-center space-y-6 p-8">
        <div className="w-24 h-24 bg-destructive/10 rounded-full flex items-center justify-center mx-auto text-destructive animate-pulse">
          <AlertTriangle size={48} />
        </div>
        
        <div className="space-y-2">
          <h1 className="text-4xl font-display font-bold text-foreground">404 Artifact Not Found</h1>
          <p className="text-muted-foreground max-w-md mx-auto">
            The coordinates you are looking for do not map to any known location in our database.
          </p>
        </div>

        <Link href="/">
          <Button size="lg" className="rounded-xl">
            Return to Base
          </Button>
        </Link>
      </div>
    </div>
  );
}
