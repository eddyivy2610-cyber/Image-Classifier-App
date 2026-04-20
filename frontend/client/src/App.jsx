import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  UploadCloud, 
  BrainCircuit, 
  CheckCircle2, 
  History, 
  Zap, 
  Loader2, 
  AlertCircle, 
  RefreshCcw, 
  Ghost,
  Fish,
  TreePine,
  Apple,
  Smartphone,
  Armchair,
  Bug,
  Users,
  Car,
  Sparkles
} from 'lucide-react';
import './App.css';

// --- Category Mapping for Icons ---
const CATEGORY_MAP = {
  'aquatic': ['beaver', 'dolphin', 'otter', 'seal', 'whale', 'aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
  'nature': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip', 'cloud', 'forest', 'mountain', 'plain', 'sea', 'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
  'food': ['bottle', 'bowl', 'can', 'cup', 'plate', 'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
  'tech': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
  'furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
  'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach', 'crab', 'lobster', 'snail', 'spider', 'worm'],
  'animals': ['bear', 'leopard', 'lion', 'tiger', 'wolf', 'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo', 'fox', 'porcupine', 'possum', 'raccoon', 'skunk', 'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
  'people': ['baby', 'boy', 'girl', 'man', 'woman'],
  'vehicles': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train', 'lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
  'ancient': ['dinosaur', 'castle', 'bridge', 'house', 'road', 'skyscraper']
};

const getIconForClass = (className) => {
  const cls = className.toLowerCase().replace(/ /g, '_');
  if (CATEGORY_MAP.aquatic.includes(cls)) return <Fish size={32} />;
  if (CATEGORY_MAP.nature.includes(cls)) return <TreePine size={32} />;
  if (CATEGORY_MAP.food.includes(cls)) return <Apple size={32} />;
  if (CATEGORY_MAP.tech.includes(cls)) return <Smartphone size={32} />;
  if (CATEGORY_MAP.furniture.includes(cls)) return <Armchair size={32} />;
  if (CATEGORY_MAP.insects.includes(cls)) return <Bug size={32} />;
  if (CATEGORY_MAP.people.includes(cls)) return <Users size={32} />;
  if (CATEGORY_MAP.vehicles.includes(cls)) return <Car size={32} />;
  return <Sparkles size={32} />;
};

const Navbar = () => (
  <nav className="navbar">
    <div className="nav-logo">
      <BrainCircuit size={20} />
    </div>
    <span className="nav-title">NeuralVision</span>
  </nav>
);

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);

  // API URL - Uses VITE_API_URL environment variable for Vercel deployment
  const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000/predict";

  const handleFile = (selectedFile) => {
    if (!selectedFile || !selectedFile.type.startsWith('image/')) {
      setError("Invalid file type. Please upload an image.");
      return;
    }
    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError(null);
  };

  const classify = async () => {
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(API_URL, { method: 'POST', body: formData });
      if (!response.ok) throw new Error("Server communication failed.");
      const data = await response.json();
      setResult(data);
      setHistory(prev => [{...data, thumbnail: preview, id: Date.now()}, ...prev].slice(0, 5));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <Navbar />
      
      <motion.main 
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className="glass-card hero-container"
      >
        <div className="header-section">
          <h1>Vision Intelligence</h1>
          <p>Harness neural networks to identify 100+ object categories in seconds.</p>
        </div>

        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div 
              key="upload"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="drop-zone"
              onClick={() => document.getElementById('fileInput').click()}
            >
              <img src="/hero.png" className="empty-state-img" alt="Hero" />
              <div className="drop-zone-content">
                <UploadCloud className="drop-icon" />
                <h3>Drop image to scan</h3>
                <p>supports JPG, PNG, WEBP</p>
              </div>
              <input type="file" id="fileInput" hidden onChange={(e) => handleFile(e.target.files[0])} />
            </motion.div>
          ) : (
            <motion.div 
              key="result"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="prediction-view"
            >
              <div className="image-preview-large">
                <img src={preview} alt="Preview" />
              </div>
              
              <div className="result-content">
                {result ? (
                  <>
                    <div className="badge">
                      <CheckCircle2 size={14} className="mr-2" />
                      <span>Analysis Complete</span>
                    </div>
                    <div className="result-header">
                      <div className="icon-wrapper">
                        {getIconForClass(result.class)}
                      </div>
                      <h2 className="class-name">{result.class}</h2>
                    </div>
                    <div>
                      <div className="confidence-label">
                        <span>Recognition Confidence</span>
                        <span className="confidence-value">{(result.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="confidence-meter">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${result.confidence * 100}%` }}
                          transition={{ duration: 1, ease: "easeOut" }}
                          className="confidence-fill" 
                        />
                      </div>
                    </div>
                    <button className="btn btn-secondary" onClick={() => { setPreview(null); setFile(null); setResult(null); }}>
                      <RefreshCcw size={18} />
                      Scan Another
                    </button>
                  </>
                ) : (
                  <>
                    <div className="header-section">
                      <h2 className="process-title">Process Image</h2>
                      <p>Ready to analyze. Optimization complete.</p>
                    </div>
                    <button className="btn btn-primary" onClick={classify} disabled={loading}>
                      {loading ? <Loader2 className="animate-spin" /> : <Zap />}
                      {loading ? 'Analyzing Neural Layers...' : 'Run Classification'}
                    </button>
                    <button className="btn btn-secondary" onClick={() => { setPreview(null); setFile(null); }}>
                      Cancel
                    </button>
                  </>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <motion.div initial={{ y: 20 }} animate={{ y: 0 }} className="error-toast">
            <AlertCircle />
            <span>{error}</span>
          </motion.div>
        )}
      </motion.main>

      <motion.aside 
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="glass-card sidebar"
      >
        <div className="history-header">
          <h3>
            <History size={18} />
            Recent Analysis
          </h3>
        </div>
        
        <div className="history-list">
          {history.length > 0 ? history.map(item => (
            <motion.div 
               key={item.id} 
               layout 
               initial={{ opacity: 0, scale: 0.9 }}
               animate={{ opacity: 1, scale: 1 }}
               className="history-item"
            >
              <img src={item.thumbnail} className="history-thumb" alt="History" />
              <div className="history-info">
                <div className="history-class">{item.class}</div>
                <div className="history-match">{(item.confidence * 100).toFixed(0)}% Match</div>
              </div>
            </motion.div>
          )) : (
            <div className="history-empty">
              <Ghost size={32} className="ghost-icon" />
              <p>No scans yet</p>
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div className="system-status">
            <div className="status-dot"></div>
            System Active
          </div>
        </div>
      </motion.aside>
    </div>
  );
}

export default App;
