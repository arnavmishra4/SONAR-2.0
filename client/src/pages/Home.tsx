import { Link } from "wouter";
import { ArrowRight, Layers, Box, Cpu } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Navigation } from "@/components/Navigation";
import { motion } from "framer-motion";

export default function Home() {
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Navigation />
      
      {/* Hero Section */}
      <section className="flex-1 flex items-center justify-center relative overflow-hidden pt-16">
        {/* Background Texture */}
        <div className="absolute inset-0 opacity-10 pointer-events-none bg-[url('https://www.transparenttextures.com/patterns/topography.png')]" />
        
        <div className="container px-4 py-24 relative z-10">
          <div className="max-w-4xl mx-auto text-center space-y-8">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              <span className="inline-block px-4 py-1.5 rounded-full border border-accent/50 bg-accent/10 text-accent-foreground text-sm font-mono tracking-wider mb-6">
                ARCHAEOLOGICAL INTELLIGENCE SYSTEM
              </span>
              <h1 className="text-5xl md:text-7xl font-display font-bold text-foreground leading-tight">
                Uncover the Past <br />
                <span className="text-primary italic">with Precision</span>
              </h1>
              <p className="text-xl text-muted-foreground mt-6 max-w-2xl mx-auto leading-relaxed">
                SONAR 2.0 leverages multi-model deep learning ensembles to detect 
                archaeological sites from LiDAR, Sentinel-2, and terrain data.
              </p>
            </motion.div>

            <motion.div 
              className="flex flex-col sm:flex-row items-center justify-center gap-4 pt-4"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <Link href="/upload">
                <Button size="lg" className="h-14 px-8 text-lg rounded-xl shadow-lg shadow-primary/20 hover:shadow-primary/40 hover:-translate-y-1 transition-all">
                  New Analysis
                  <ArrowRight className="ml-2 w-5 h-5" />
                </Button>
              </Link>
              <Link href="/dashboard">
                <Button size="lg" variant="outline" className="h-14 px-8 text-lg rounded-xl bg-background/50 backdrop-blur hover:bg-background/80">
                  View Demo Data
                </Button>
              </Link>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="bg-muted/30 border-t border-border py-24">
        <div className="container px-4">
          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard 
              icon={Layers}
              title="Multi-Modal Input"
              description="Ingest LiDAR DTM, Sentinel-2 imagery, and HydroSHEDS data simultaneously for comprehensive site analysis."
            />
            <FeatureCard 
              icon={Cpu}
              title="Ensemble AI Models"
              description="Combines Autoencoders, Isolation Forests, K-Means, and GATE architecture for robust anomaly detection."
            />
            <FeatureCard 
              icon={Box}
              title="3D Visualization"
              description="Interactive heatmap generation and topographical analysis to pinpoint potential excavation sites."
            />
          </div>
        </div>
      </section>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, description }: { icon: any, title: string, description: string }) {
  return (
    <div className="bg-background border border-border/50 p-8 rounded-2xl shadow-sm hover:shadow-md transition-shadow group">
      <div className="w-12 h-12 bg-primary/10 rounded-xl flex items-center justify-center text-primary mb-6 group-hover:scale-110 transition-transform duration-300">
        <Icon size={24} />
      </div>
      <h3 className="text-xl font-display font-bold mb-3 text-foreground">{title}</h3>
      <p className="text-muted-foreground leading-relaxed">
        {description}
      </p>
    </div>
  );
}
