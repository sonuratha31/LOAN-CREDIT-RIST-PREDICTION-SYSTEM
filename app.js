document.addEventListener('DOMContentLoaded', async () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('actual-result');
    const scoreText = document.getElementById('score-text');
    const probFill = document.getElementById('prob-fill');
    const statusBadge = document.getElementById('status-badge');
    const historyTable = document.getElementById('history-table').querySelector('tbody');
    const exportBtn = document.getElementById('export-btn');
    
    let predictionsHistory = [];

    // Stability elements
    const cvMean = document.getElementById('cv-mean');
    const cvStd = document.getElementById('cv-std');
    const valTN = document.getElementById('val-tn');
    const valFP = document.getElementById('val-fp');
    const valFN = document.getElementById('val-fn');
    const valTP = document.getElementById('val-tp');
    
    // Heatmap container
    const heatmapDiv = document.getElementById('heatmap');

    // --- Stats & Metrics Loader ---
    async function loadStats() {
        try {
            const resp = await fetch('/stats');
            const data = await resp.json();
            renderImportanceChart(data.top_features, data.top_importances);
            if (data.metrics) {
                const m = data.metrics;
                cvMean.textContent = (m.cv_f1_mean * 100).toFixed(1) + '%';
                cvStd.textContent = (m.cv_f1_std * 100).toFixed(1) + '%';
                const cm = m.confusion_matrix;
                valTN.textContent = cm[0][0];
                valFP.textContent = cm[0][1];
                valFN.textContent = cm[1][0];
                valTP.textContent = cm[1][1];
                renderHeatmap(m.correlation_data);
            }
        } catch (err) {
            console.error('Failed to load dashboard metrics:', err);
        }
    }

    function renderImportanceChart(labels, values) {
        const ctx = document.getElementById('importanceChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels.map(l => l.replace(/_/g, ' ').toUpperCase()),
                datasets: [{
                    label: 'Decision Power',
                    data: values,
                    backgroundColor: 'rgba(79, 70, 229, 0.4)',
                    borderColor: '#4F46E5',
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                scales: {
                    x: { beginAtZero: true, grid: { color: '#334155' }, ticks: { display: false } },
                    y: { grid: { display: false }, ticks: { color: '#94A3B8', font: { size: 9 } } }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    function renderHeatmap(corrData) {
        heatmapDiv.innerHTML = '';
        const values = corrData.values;
        const columns = corrData.columns;
        values.forEach((row, rIdx) => {
            row.forEach((val, cIdx) => {
                const cell = document.createElement('div');
                cell.className = 'heatmap-cell';
                const hue = val > 0 ? 120 : 0; 
                const opacity = Math.abs(val);
                cell.style.backgroundColor = `hsla(${hue}, 100%, 50%, ${opacity})`;
                cell.title = `${columns[rIdx]} vs ${columns[cIdx]}: ${val}`;
                heatmapDiv.appendChild(cell);
            });
        });
    }

    // --- History Management ---
    function addToHistory(inputData, result) {
        const prediction = {
            timestamp: new Date().toLocaleTimeString(),
            age: inputData.age,
            job: inputData.job,
            probability: (result.probability * 100).toFixed(1),
            status: result.status
        };
        
        predictionsHistory.unshift(prediction); // Add to start
        if (predictionsHistory.length > 5) predictionsHistory.pop(); // Keep last 5 for UI

        updateHistoryTable();
    }

    function updateHistoryTable() {
        historyTable.innerHTML = '';
        predictionsHistory.forEach(p => {
            const row = `<tr>
                <td>${p.age}</td>
                <td>${p.job}</td>
                <td>${p.probability}%</td>
                <td style="color: ${p.status === 'Eligible' ? '#10B981' : '#F87171'}">${p.status}</td>
            </tr>`;
            historyTable.innerHTML += row;
        });
    }

    // --- CSV Export Logic ---
    exportBtn.addEventListener('click', () => {
        if (predictionsHistory.length === 0) {
            alert('No predictions to export yet.');
            return;
        }

        const headers = ['Timestamp', 'Age', 'Job', 'Probability', 'Result'];
        const rows = predictionsHistory.map(p => [p.timestamp, p.age, p.job, p.probability, p.status]);
        
        let csvContent = "data:text/csv;charset=utf-8," 
            + headers.join(",") + "\n"
            + rows.map(e => e.join(",")).join("\n");

        const encodedUri = encodeURI(csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `loan_predictions_${new Date().toISOString().slice(0,10)}.csv`);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });

    // --- Prediction Request ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        statusBadge.textContent = 'ANALYZING...';
        statusBadge.className = 'status-badge';
        scoreText.textContent = '---';
        probFill.style.width = '0%';

        const formData = new FormData(form);
        const data = Object.fromEntries(formData);
        ['age', 'campaign', 'pdays', 'previous'].forEach(k => data[k] = parseFloat(data[k]));

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            
            setTimeout(() => {
                const probPercent = (result.probability * 100).toFixed(1);
                scoreText.textContent = probPercent + '%';
                probFill.style.width = probPercent + '%';
                
                statusBadge.textContent = result.status;
                statusBadge.className = 'status-badge ' + (result.loan_approved === 1 ? 'status-approved' : 'status-rejected');
                
                addToHistory(data, result);
            }, 500);
        } catch (err) {
            console.error('Prediction failed:', err);
            statusBadge.textContent = 'ERROR';
        }
    });

    loadStats();
});
