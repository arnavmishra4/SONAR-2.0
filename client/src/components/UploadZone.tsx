import { useCallback, useState } from 'react';
import { UploadCloud, File, CheckCircle2 } from 'lucide-react';
import { Card } from '@/components/ui/card';

interface UploadZoneProps {
  label: string;
  acceptedFileTypes?: string;
  onFileSelect?: (file: File) => void;
  required?: boolean;
}

export function UploadZone({ label, acceptedFileTypes = ".tif,.tiff,.asc", onFileSelect, required }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.length) {
      const droppedFile = e.dataTransfer.files[0];
      setFile(droppedFile);
      onFileSelect?.(droppedFile);
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      onFileSelect?.(selectedFile);
    }
  }, [onFileSelect]);

  return (
    <div className="relative group">
      <input
        type="file"
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
        accept={acceptedFileTypes}
        onChange={handleFileInput}
      />
      <Card 
        className={`
          relative overflow-hidden transition-all duration-300 border-2 border-dashed
          flex flex-col items-center justify-center p-8 text-center min-h-[200px]
          ${isDragging 
            ? 'border-accent bg-accent/5 scale-[1.02]' 
            : file 
              ? 'border-primary bg-primary/5' 
              : 'border-muted-foreground/20 hover:border-primary/50 hover:bg-muted/50'
          }
        `}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {file ? (
          <div className="flex flex-col items-center gap-3 animate-in fade-in zoom-in duration-300">
            <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center text-primary">
              <CheckCircle2 size={24} />
            </div>
            <div>
              <p className="font-medium text-foreground">{file.name}</p>
              <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4">
            <div className={`
              w-16 h-16 rounded-2xl flex items-center justify-center transition-colors
              ${isDragging ? 'bg-accent text-accent-foreground' : 'bg-muted text-muted-foreground group-hover:bg-primary/10 group-hover:text-primary'}
            `}>
              <UploadCloud size={32} />
            </div>
            <div className="space-y-1">
              <p className="font-display font-bold text-lg text-foreground">
                {label} {required && <span className="text-destructive">*</span>}
              </p>
              <p className="text-sm text-muted-foreground max-w-[200px] mx-auto">
                Drag & drop or click to upload
              </p>
            </div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-background border text-xs font-mono text-muted-foreground">
              <File size={12} />
              {acceptedFileTypes.replace(/,/g, ' ')}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
}
