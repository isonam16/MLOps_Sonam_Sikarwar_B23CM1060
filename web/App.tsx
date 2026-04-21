
import React, { useState } from 'react';
import { AppFeature } from './types';
import FeatureSelector from './components/FeatureSelector';
import ImageAnalyzer from './components/ImageAnalyzer';
import AdvancedGenerator from './components/AdvancedGenerator';
import { Code2 } from 'lucide-react';

const App: React.FC = () => {
  const [currentFeature, setCurrentFeature] = useState<AppFeature>(AppFeature.IMAGE_ANALYSIS);

  return (
    <div className="h-screen bg-black text-zinc-200 flex flex-col overflow-hidden font-sans selection:bg-blue-500/30">
      
      {/* Top Navigation Bar */}
      <header className="h-14 border-b border-zinc-900 bg-black/50 backdrop-blur-md flex items-center px-6 justify-between shrink-0 z-50">
        <div className="flex items-center gap-6">
          <FeatureSelector currentFeature={currentFeature} onSelect={setCurrentFeature} />
        </div>
        
        <div className="flex items-center gap-4 text-sm text-zinc-500">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
            System Online
          </span>
        </div>
      </header>

      {/* Main Content Workspace */}
      <main className="flex-1 overflow-hidden relative">
         <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,_var(--tw-gradient-stops))] from-blue-900/10 via-transparent to-transparent pointer-events-none" />
         
         <div className="h-full relative z-10 p-4">
            {currentFeature === AppFeature.IMAGE_ANALYSIS && (
              <ImageAnalyzer />
            )}
            
            {currentFeature === AppFeature.ADVANCED_GENERATOR && (
              <AdvancedGenerator />
            )}
         </div>
      </main>
      
    </div>
  );
};

export default App;
