import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

// 🚀 UPDATE THIS TO YOUR NEW CUSTOM LOCALTUNNEL URL!
const BASE_URL = "https://bitter-dogs-call.loca.lt"; 

const SAMPLE_DATA = [
  { id: "VIR-01", family: "Viral Spike", sequence: "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLH" },
  { id: "VIR-02", family: "Viral Spike", sequence: "MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLH" },
  { id: "VIR-03", family: "Viral Spike", sequence: "MYSFVSEETGTLIVNSVLLFLAFVVFLLVTLAILTALRLCAYCCNIVNV" },
  { id: "KIN-01", family: "Human Kinase", sequence: "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAI" },
  { id: "KIN-02", family: "Human Kinase", sequence: "MSGALLLPRAREAGVGGGVGAVEGEGAAPVPPPGPAGPAPAAAAAPAEAA" },
  { id: "KIN-03", family: "Human Kinase", sequence: "MEQQCQAEARVKGLLQLAAQRLLQQRLQAAERQQQKQQQQQQQQQQQQQ" },
  { id: "ANTI-01", family: "Antibody", sequence: "QVQLQESGPGLVRPSQTLSLTCTVSGSTFSNDYYTWVRQPPGRGLEWIGY" },
  { id: "ANTI-02", family: "Antibody", sequence: "EVQLVESGGGLVQPGGSLRLSCAASGFTFTNYAMNWVRQAPGKGLEWVAR" },
  { id: "ANTI-03", family: "Antibody", sequence: "DIQMTQSPSSLSASVGDRVTITCRASQGISNYLAWYQQKPGKVPKLLIYAA" }
];

const FAMILY_COLORS = { "Viral Spike": "#EF4444", "Human Kinase": "#3B82F6", "Antibody": "#10B981" };

export default function BioCudaDashboard() {
  const [activeTab, setActiveTab] = useState('single');
  
  const [sequence, setSequence] = useState("MALWMRLLPLLALLALWGPDPAAA");
  const [embedding, setEmbedding] = useState(null);
  const [plotData, setPlotData] = useState([]);
  
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const runSingleExtraction = async () => {
    setLoading(true); setError(""); setEmbedding(null);
    try {
      const res = await fetch(`${BASE_URL}/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Bypass-Tunnel-Reminder": "true" },
        body: JSON.stringify({ sequence })
      });
      if (!res.ok) throw new Error("Backend failed. Is the API and Tunnel running?");
      setEmbedding(await res.json());
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };

  const runClustering = async () => {
    setLoading(true); setError("");
    try {
      const res = await fetch(`${BASE_URL}/cluster`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Bypass-Tunnel-Reminder": "true" },
        body: JSON.stringify({ proteins: SAMPLE_DATA })
      });
      if (!res.ok) throw new Error("Backend failed. Is the API and Tunnel running?");
      const data = await res.json();
      setPlotData(data.clusters);
    } catch (err) { setError(err.message); } finally { setLoading(false); }
  };

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div style={{ backgroundColor: '#fff', border: '3px solid #000', padding: '15px', boxShadow: '4px 4px 0px #000' }}>
          <p><strong>ID:</strong> {data.id}</p>
          <p><strong>Family:</strong> {data.family}</p>
        </div>
      );
    }
    return null;
  };

  const tabStyle = (isActive) => ({
    padding: '10px 20px', fontWeight: 'bold', fontSize: '16px', cursor: 'pointer',
    backgroundColor: isActive ? '#000' : '#fff', color: isActive ? '#fff' : '#000',
    border: '3px solid #000', borderBottom: isActive ? 'none' : '3px solid #000',
    textTransform: 'uppercase'
  });

  return (
    <div style={{ padding: '40px', fontFamily: 'monospace', backgroundColor: '#e5e5f7', minHeight: '100vh', backgroundImage: 'radial-gradient(#000 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
      <div style={{ maxWidth: '1000px', margin: '0 auto', backgroundColor: '#fff', border: '4px solid #000', boxShadow: '8px 8px 0px #000', padding: '30px' }}>
        
        <h1 style={{ margin: 0, textTransform: 'uppercase', marginBottom: '10px' }}>
          🧬 BioCUDA Latent Explorer
        </h1>

        {/* PORTFOLIO SLEEP MODE WARNING */}
<div style={{ backgroundColor: '#FEF08A', border: '2px solid #CA8A04', padding: '15px', marginBottom: '30px', color: '#854D0E', fontWeight: 'bold', fontSize: '14px' }}>
  ⚠️ CLOUD COMPUTE STATUS: OFFLINE <br/><br/>
  To conserve enterprise GPU costs, the PyTorch A100/T4 backend API is currently in sleep mode. The dashboard UI is fully functional, but live inference is paused. Please follow the repo instructions to boot the API locally.
</div>
        
        {/* HARDWARE STATUS BADGE */}
        <div style={{ backgroundColor: '#D4E157', border: '2px solid #000', padding: '10px', marginBottom: '30px', display: 'inline-block', fontWeight: 'bold' }}>
          🔌 ACTIVE PIPELINE: STABILITY BRANCH (T4 Edge API / 8-Bit Quantized)
        </div>

        <div style={{ display: 'flex', gap: '10px', marginBottom: '30px', borderBottom: '4px solid #000' }}>
          <button style={tabStyle(activeTab === 'single')} onClick={() => setActiveTab('single')}>Single Sequence Heatmap</button>
          <button style={tabStyle(activeTab === 'cluster')} onClick={() => setActiveTab('cluster')}>Batch AI Clustering</button>
        </div>

        {error && <div style={{ backgroundColor: '#FECACA', border: '2px solid #DC2626', padding: '10px', color: '#DC2626', fontWeight: 'bold', marginBottom: '20px' }}>🚨 ERROR: {error}</div>}

        {/* --- TAB 1: SINGLE EXTRACTION --- */}
        {activeTab === 'single' && (
          <div>
            <label style={{ fontWeight: 'bold', display: 'block', marginBottom: '10px' }}>INPUT AMINO ACID SEQUENCE:</label>
            <textarea 
              value={sequence} onChange={(e) => setSequence(e.target.value.toUpperCase())}
              style={{ width: '100%', height: '100px', border: '2px solid #000', padding: '10px', fontSize: '16px', fontFamily: 'monospace', resize: 'vertical' }}
            />
            <button onClick={runSingleExtraction} disabled={loading}
              style={{ marginTop: '15px', padding: '12px 24px', fontWeight: 'bold', fontSize: '16px', backgroundColor: loading ? '#ccc' : '#000', color: loading ? '#666' : '#fff', border: '3px solid #000', cursor: loading ? 'wait' : 'pointer', boxShadow: loading ? 'none' : '4px 4px 0px #D4E157', textTransform: 'uppercase', transition: 'all 0.2s' }}>
              {loading ? "Extracting..." : "Generate Embedding"}
            </button>

            {embedding && (
              <div style={{ marginTop: '40px', borderTop: '4px solid #000', paddingTop: '20px' }}>
                <h2 style={{ textTransform: 'uppercase', margin: '0 0 10px 0' }}>Extraction Successful</h2>
                <p style={{ fontSize: '16px', fontWeight: 'bold', margin: '5px 0' }}>Dimensions Extracted: {embedding.embedding_dimensions}</p>
                <p style={{ fontSize: '16px', color: '#666', margin: '5px 0 15px 0' }}>Vector Preview (First 50 dimensions):</p>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(10, 1fr)', gap: '5px' }}>
                  {embedding.vector.slice(0, 50).map((val, i) => (
                    <div key={i} style={{ backgroundColor: val > 0 ? `rgba(16, 185, 129, ${val * 5})` : `rgba(239, 68, 68, ${Math.abs(val) * 5})`, height: '25px', border: '1px solid #000' }} title={`Dim ${i}: ${val.toFixed(4)}`} />
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* --- TAB 2: CLUSTERING --- */}
        {activeTab === 'cluster' && (
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <button onClick={runClustering} disabled={loading}
                style={{ padding: '12px 24px', fontWeight: 'bold', fontSize: '16px', backgroundColor: loading ? '#ccc' : '#000', color: loading ? '#fff' : '#fff', border: '3px solid #000', cursor: loading ? 'wait' : 'pointer', boxShadow: loading ? 'none' : '4px 4px 0px #D4E157', textTransform: 'uppercase' }}>
                {loading ? "Crunching Batch..." : "Run AI Clustering"}
              </button>
              
              <div style={{ display: 'flex', gap: '15px' }}>
                {Object.entries(FAMILY_COLORS).map(([family, color]) => (
                  <div key={family} style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 'bold', fontSize: '14px' }}>
                    <div style={{ width: '16px', height: '16px', backgroundColor: color, border: '2px solid #000' }} />{family}
                  </div>
                ))}
              </div>
            </div>

            <div style={{ height: '500px', border: '4px solid #000', backgroundColor: '#fff', position: 'relative', backgroundImage: 'linear-gradient(#f0f0f0 1px, transparent 1px), linear-gradient(90deg, #f0f0f0 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
              {plotData.length === 0 && !loading && (<div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', fontWeight: 'bold', fontSize: '20px', color: '#666' }}>AWAITING DATA...</div>)}
              {plotData.length > 0 && (
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ccc" />
                    <XAxis type="number" dataKey="x" name="PCA Dimension 1" tick={false} axisLine={{ stroke: '#000', strokeWidth: 2 }} />
                    <YAxis type="number" dataKey="y" name="PCA Dimension 2" tick={false} axisLine={{ stroke: '#000', strokeWidth: 2 }} />
                    <Tooltip content={<CustomTooltip />} cursor={{ strokeDasharray: '3 3' }} />
                    <Scatter data={plotData} fill="#000">
                      {plotData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={FAMILY_COLORS[entry.family]} stroke="#000" strokeWidth={2} />
                      ))}
                    </Scatter>
                  </ScatterChart>
                </ResponsiveContainer>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}