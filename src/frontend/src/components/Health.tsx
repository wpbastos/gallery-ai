import { useState, useEffect } from 'react';
import axios from 'axios';
import './Health.css';

interface HealthStatus {
  status: string;
  message: string;
}

function Health() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/health');
        setHealth(response.data);
        setError(null);
      } catch (err) {
        setError('Failed to fetch health status');
        console.error('Error fetching health status:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();
  }, []);

  if (loading) {
    return <div>Loading health status...</div>;
  }

  if (error) {
    return <div className="error">{error}</div>;
  }

  return (
    <div className="health-status">
      <h2>API Health Status</h2>
      {health && (
        <div>
          <p>Status: <span className={`status ${health.status}`}>{health.status}</span></p>
          <p>Message: {health.message}</p>
        </div>
      )}
    </div>
  );
}

export default Health; 