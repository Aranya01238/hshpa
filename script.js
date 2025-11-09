// ========================
// STATE MANAGEMENT
// ========================
let appState = {
    rawData: [],
    headers: [],
    model: null,
    perfChart: null,
    predictions: [],
};

// ========================
// UTILITY FUNCTIONS
// ========================

function showToast(message, duration = 3000) {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => toast.classList.remove('show'), duration);
}

function showStatus(elementId, message, type = 'loading') {
    const statusEl = document.getElementById(elementId);
    statusEl.textContent = message;
    statusEl.className = `status-message show ${type}`;
}

function clearStatus(elementId) {
    const statusEl = document.getElementById(elementId);
    statusEl.classList.remove('show');
}

function meanStd(arr) {
    const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
    const variance = arr.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / arr.length;
    const std = Math.sqrt(variance) || 1;
    return { mean, std };
}

function rSquared(yTrue, yPred) {
    const meanY = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
    const ssTot = yTrue.reduce((s, y) => s + Math.pow(y - meanY, 2), 0);
    const ssRes = yTrue.reduce((s, y, i) => s + Math.pow(y - yPred[i], 2), 0);
    return 1 - (ssRes / ssTot);
}

function rmse(yTrue, yPred) {
    const mse = yTrue.reduce((s, y, i) => s + Math.pow(y - yPred[i], 2), 0) / yTrue.length;
    return Math.sqrt(mse);
}

// ========================
// FILE UPLOAD HANDLER
// ========================

const uploadArea = document.getElementById('uploadArea');
const csvFile = document.getElementById('csvFile');

uploadArea.addEventListener('click', () => csvFile.click());

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = 'rgba(0, 234, 255, 0.15)';
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = 'rgba(0, 234, 255, 0.03)';
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        csvFile.files = files;
        handleFileUpload({ target: { files } });
    }
});

csvFile.addEventListener('change', handleFileUpload);

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    showStatus('fileStatus', '⏳ Parsing CSV file...', 'loading');

    // Declare Papa variable
    const Papa = window.Papa;

    Papa.parse(file, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true,
        worker: true,
        complete: (res) => {
            appState.rawData = res.data.filter(row => 
                Object.values(row).some(v => v !== null && v !== '' && v !== undefined)
            );
            appState.headers = res.meta.fields || [];
            
            clearStatus('fileStatus');
            showStatus('fileStatus', `✅ Loaded ${appState.rawData.length} rows and ${appState.headers.length} columns`, 'success');
            showToast(`Dataset loaded: ${appState.rawData.length} rows`);
        },
        error: (err) => {
            showStatus('fileStatus', `❌ Error: ${err.message}`, 'error');
        }
    });
}

// ========================
// MODEL TRAINING
// ========================

document.getElementById('trainBtn').addEventListener('click', trainModel);

function trainModel() {
    if (!appState.rawData.length) {
        showToast('Please upload a CSV file first');
        return;
    }

    showStatus('trainingStatus', '🚀 Training model...', 'loading');

    setTimeout(() => {
        try {
            // Detect numeric columns
            const numericCols = appState.headers.filter(h => 
                appState.rawData.some(r => {
                    const v = parseFloat(r[h]);
                    return isFinite(v);
                })
            );

            if (!numericCols.length) {
                throw new Error('No numeric columns found');
            }

            // Find target column
            const target = appState.headers.find(h => 
                h.toLowerCase().includes('price') || h.toLowerCase().includes('target')
            ) || numericCols[numericCols.length - 1];

            const features = numericCols.filter(h => h !== target);

            if (!features.length) {
                throw new Error('Not enough features for training');
            }

            // Clean data
            const clean = appState.rawData.map(r => {
                const obj = {};
                numericCols.forEach(c => {
                    const v = parseFloat(r[c]);
                    obj[c] = isFinite(v) ? v : 0;
                });
                return obj;
            }).filter(r => Object.values(r).some(v => v !== 0));

            if (clean.length < 5) {
                throw new Error('Not enough valid data points (need >= 5)');
            }

            // Prepare training data
            const X = clean.map(r => features.map(f => r[f]));
            const Y = clean.map(r => r[target]);
            const n = X.length;

            // Compute statistics
            const featureStats = features.map((f, j) => {
                const col = X.map(row => row[j]);
                return meanStd(col);
            });
            const targetStats = meanStd(Y);

            // Standardize
            const Xs = X.map(row => 
                row.map((v, j) => (v - featureStats[j].mean) / featureStats[j].std)
            );
            const Ys = Y.map(v => (v - targetStats.mean) / targetStats.std);

            // Train linear regression
            const weights = features.map((_, j) => {
                let num = 0, den = 0;
                for (let i = 0; i < n; i++) {
                    num += Xs[i][j] * Ys[i];
                    den += Xs[i][j] * Xs[i][j];
                }
                return num / (den || 1);
            });

            // Store model
            appState.model = {
                features,
                target,
                weights,
                featureStats,
                targetStats,
                n
            };

            // Evaluate
            const yPredStd = Xs.map(row => 
                weights.reduce((s, w, j) => s + w * row[j], 0)
            );
            const yPred = yPredStd.map(v => v * targetStats.std + targetStats.mean);

            const r2 = rSquared(Y, yPred);
            const error = rmse(Y, yPred);

            // Display results
            buildInputs(features);
            updateChart(r2, error);
            displayMetrics(r2, error, n);
            displayInsights(clean, features, target);

            clearStatus('trainingStatus');
            showStatus('trainingStatus', `✅ Model trained! R² = ${r2.toFixed(3)}, RMSE = ${error.toFixed(2)}`, 'success');
            showToast('Model trained successfully!');
        } catch (err) {
            clearStatus('trainingStatus');
            showStatus('trainingStatus', `❌ Error: ${err.message}`, 'error');
            showToast(`Error: ${err.message}`);
        }
    }, 500);
}

// ========================
// BUILD PREDICTION INPUTS
// ========================

function buildInputs(features) {
    const container = document.getElementById('predictInputs');
    container.innerHTML = '';
    features.forEach(f => {
        const group = document.createElement('div');
        group.className = 'input-group';
        group.innerHTML = `
            <label>${f}</label>
            <input type="number" id="in-${f}" placeholder="Enter ${f}" step="any">
        `;
        container.appendChild(group);
    });
}

// ========================
// DISPLAY METRICS
// ========================

function displayMetrics(r2, rmse, n) {
    const metricsEl = document.getElementById('metricsText');
    metricsEl.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
            <div class="insight-item">
                <div class="insight-label">R² Score</div>
                <div class="insight-value">${(r2 * 100).toFixed(1)}%</div>
            </div>
            <div class="insight-item">
                <div class="insight-label">RMSE</div>
                <div class="insight-value">${rmse.toFixed(0)}</div>
            </div>
            <div class="insight-item">
                <div class="insight-label">Samples</div>
                <div class="insight-value">${n}</div>
            </div>
        </div>
    `;
    metricsEl.classList.add('show');
}

function displayInsights(clean, features, target) {
    const insightsEl = document.getElementById('insightsDisplay');
    const targetValues = clean.map(r => r[target]);
    const targetStats = meanStd(targetValues);

    let html = `
        <div class="insight-item">
            <div class="insight-label">${target} (Mean)</div>
            <div class="insight-value">$${targetStats.mean.toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
        </div>
        <div class="insight-item">
            <div class="insight-label">${target} (Std Dev)</div>
            <div class="insight-value">$${targetStats.std.toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
        </div>
    `;

    features.slice(0, 2).forEach(f => {
        const fValues = clean.map(r => r[f]);
        const fStats = meanStd(fValues);
        html += `
            <div class="insight-item">
                <div class="insight-label">${f} (Mean)</div>
                <div class="insight-value">${fStats.mean.toLocaleString('en-US', { maximumFractionDigits: 2 })}</div>
            </div>
        `;
    });

    insightsEl.innerHTML = html;
}

// ========================
// CHART VISUALIZATION
// ========================

function updateChart(r2, rmse) {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    if (appState.perfChart) {
        appState.perfChart.destroy();
    }

    // Declare Chart variable
    const Chart = window.Chart;

    appState.perfChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['R² Score', 'RMSE'],
            datasets: [{
                label: 'Model Metrics',
                data: [r2 * 100, rmse / 1000], // Normalize RMSE for visualization
                backgroundColor: ['#00ffcc', '#ff6666'],
                borderRadius: 8,
                borderSkipped: false
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    grid: {
                        display: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

// ========================
// PREDICTION
// ========================

document.getElementById('predictBtn').addEventListener('click', predict);

function predict() {
    if (!appState.model) {
        showToast('Please train the model first');
        return;
    }

    try {
        // Get inputs
        const inputsRaw = appState.model.features.map(f => {
            const val = parseFloat(document.getElementById(`in-${f}`).value) || 0;
            return val;
        });

        // Standardize
        const inputsStd = inputsRaw.map((v, j) => 
            (v - appState.model.featureStats[j].mean) / appState.model.featureStats[j].std
        );

        // Predict
        const yStd = appState.model.weights.reduce((s, w, j) => s + w * inputsStd[j], 0);
        const y = yStd * appState.model.targetStats.std + appState.model.targetStats.mean;

        // Display prediction
        const outputEl = document.getElementById('predictionOutput');
        outputEl.innerHTML = `
            <div class="price-label">Predicted ${appState.model.target}</div>
            <div class="price-tag">$${y.toLocaleString('en-US', { maximumFractionDigits: 0 })}</div>
        `;
        outputEl.classList.add('show');

        // Detailed report
        const reportEl = document.getElementById('detailedReport');
        const deviation = (y - appState.model.targetStats.mean) / appState.model.targetStats.mean * 100;
        const segment = deviation < -10 ? 'Budget' : deviation > 10 ? 'Luxury' : 'Mid-Range';

        reportEl.innerHTML = `
            <h4>📊 Prediction Report</h4>
            <div class="report-item">
                <span class="report-label">Predicted Value</span>
                <span class="report-value">$${y.toLocaleString('en-US', { maximumFractionDigits: 0 })}</span>
            </div>
            <div class="report-item">
                <span class="report-label">vs. Dataset Mean</span>
                <span class="report-value">${deviation >= 0 ? '+' : ''}${deviation.toFixed(1)}%</span>
            </div>
            <div class="report-item">
                <span class="report-label">Market Segment</span>
                <span class="report-value">${segment}</span>
            </div>
            <div class="report-item">
                <span class="report-label">Model Confidence</span>
                <span class="report-value">${(appState.model.confidence * 100).toFixed(0)}%</span>
            </div>
            <div class="report-item">
                <span class="report-label">Used Features</span>
                <span class="report-value">${appState.model.features.length}</span>
            </div>
        `;
        reportEl.classList.add('show');

        // Store prediction
        appState.predictions.push({ inputs: inputsRaw, prediction: y, timestamp: new Date() });

        showToast('Prediction complete!');
    } catch (err) {
        showToast(`Error: ${err.message}`);
    }
}

// ========================
// EXPORT PDF
// ========================

document.getElementById('exportBtn').addEventListener('click', exportPDF);

function exportPDF() {
    if (!appState.model) {
        showToast('No predictions to export');
        return;
    }

    const element = document.getElementById('detailedReport');
    if (!element.classList.contains('show')) {
        showToast('Please make a prediction first');
        return;
    }

    const opt = {
        margin: 10,
        filename: 'prediction-report.pdf',
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2 },
        jsPDF: { orientation: 'portrait', unit: 'mm', format: 'a4' }
    };

    // Declare html2pdf variable
    const html2pdf = window.html2pdf;

    html2pdf().set(opt).from(element).save();
    showToast('PDF exported successfully!');
}

// ========================
// NAVIGATION
// ========================

document.getElementById('scrollToDemo').addEventListener('click', () => {
    document.getElementById('demo').scrollIntoView({ behavior: 'smooth' });
});

// ========================
// THEME TOGGLE
// ========================

const themeToggle = document.getElementById('themeToggle');

themeToggle.addEventListener('click', () => {
    document.body.classList.toggle('light-mode');
    const isDark = !document.body.classList.contains('light-mode');
    themeToggle.textContent = isDark ? '🌙' : '☀️';
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
});

// Load saved theme
if (localStorage.getItem('theme') === 'light') {
    document.body.classList.add('light-mode');
    themeToggle.textContent = '☀️';
}

// ========================
// INITIALIZATION
// ========================

// Calculate model confidence after training
function calculateConfidence() {
    if (appState.model) {
        appState.model.confidence = Math.min(0.95, Math.max(0.5, 0.5 + (appState.model.n / 1000)));
    }
}

// Initial setup
document.addEventListener('DOMContentLoaded', () => {
    console.log('[v0] App initialized successfully');
});