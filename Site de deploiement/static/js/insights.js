// Global Insights Page JavaScript
document.addEventListener('DOMContentLoaded', () => {
    initializeInsights();
});

function initializeInsights() {
    // Count tasks
    let classificationCount = 0;
    let regressionCount = 0;
    
    for (const dataset in DATASETS_INFO) {
        if (DATASETS_INFO[dataset].task_type === 'classification') {
            classificationCount++;
        } else {
            regressionCount++;
        }
    }
    
    document.getElementById('classification-count').textContent = classificationCount;
    document.getElementById('regression-count').textContent = regressionCount;
    
    // Initialize charts
    createPerformanceChart();
    createComplexityChart();
    createPerformanceTable();
    createWinRateChart();
    loadROCCurves();
    createPerformanceDistributions();
    createPerformanceHeatmap();
}

function createPerformanceChart() {
    const ctx = document.getElementById('performanceChart').getContext('2d');
    
    const datasets = [];
    const models = ['RLT', 'Random Forest', 'Gradient Boosting', 'ExtraTrees', 'Lasso'];
    const colors = [
        'rgba(91, 33, 182, 0.8)',   // Primary purple
        'rgba(5, 150, 105, 0.8)',    // Secondary green
        'rgba(6, 182, 212, 0.8)',    // Accent cyan
        'rgba(239, 68, 68, 0.8)',    // Danger red
        'rgba(245, 158, 11, 0.8)'    // Warning orange
    ];
    
    const datasetNames = Object.keys(PERFORMANCE_DATA);
    
    models.forEach((model, idx) => {
        const data = datasetNames.map(dataset => {
            const perf = PERFORMANCE_DATA[dataset];
            if (!perf || !perf.models || !perf.models[model]) return null;
            
            // Use accuracy for classification, R² for regression
            if (perf.task_type === 'classification') {
                return perf.models[model].accuracy * 100;
            } else {
                return perf.models[model].r2 * 100;
            }
        });
        
        datasets.push({
            label: model,
            data: data,
            backgroundColor: colors[idx],
            borderColor: colors[idx].replace('0.8', '1'),
            borderWidth: 2
        });
    });
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: datasetNames,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Across Datasets (Accuracy% / R²%)',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'top',
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Performance (%)'
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45,
                        minRotation: 45
                    }
                }
            }
        }
    });
}

function createComplexityChart() {
    const ctx = document.getElementById('complexityChart').getContext('2d');
    
    const datasetNames = [];
    const samples = [];
    const features = [];
    
    for (const dataset in DATASETS_INFO) {
        datasetNames.push(dataset);
        samples.push(DATASETS_INFO[dataset].n_samples);
        features.push(DATASETS_INFO[dataset].n_features);
    }
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Datasets',
                data: datasetNames.map((name, idx) => ({
                    x: features[idx],
                    y: samples[idx],
                    label: name
                })),
                backgroundColor: 'rgba(91, 33, 182, 0.6)',
                borderColor: 'rgba(91, 33, 182, 1)',
                borderWidth: 2,
                pointRadius: 8,
                pointHoverRadius: 12
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Dataset Complexity (Features vs Samples)',
                    font: { size: 16, weight: 'bold' }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `${context.raw.label}: ${context.parsed.x} features, ${context.parsed.y} samples`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Number of Features'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Number of Samples'
                    }
                }
            }
        }
    });
}

function createPerformanceTable() {
    const tbody = document.getElementById('performance-table-body');
    tbody.innerHTML = '';
    
    for (const dataset in PERFORMANCE_DATA) {
        const perf = PERFORMANCE_DATA[dataset];
        const row = document.createElement('tr');
        
        const taskBadge = perf.task_type === 'classification'
            ? '<span class="dataset-badge dataset-badge-classification">Classification</span>'
            : '<span class="dataset-badge dataset-badge-regression">Regression</span>';
        
        let cells = `<td><strong>${dataset}</strong></td><td>${taskBadge}</td>`;
        
        const models = ['RLT', 'Random Forest', 'Gradient Boosting', 'ExtraTrees'];
        
        // Add Lasso/Ridge based on task type
        if (perf.task_type === 'classification') {
            models.push('Logistic Ridge');
        } else {
            models.push('Lasso');
        }
        
        models.forEach(model => {
            if (perf.models && perf.models[model]) {
                const modelPerf = perf.models[model];
                let value = '';
                
                if (perf.task_type === 'classification') {
                    value = `${(modelPerf.accuracy * 100).toFixed(2)}%`;
                } else {
                    value = `R²: ${(modelPerf.r2 * 100).toFixed(2)}%`;
                }
                
                cells += `<td style="text-align: center;">${value}</td>`;
            } else {
                cells += `<td style="text-align: center; color: var(--text-light);">-</td>`;
            }
        });
        
        row.innerHTML = cells;
        tbody.appendChild(row);
    }
}

function createWinRateChart() {
    const ctx = document.getElementById('winRateChart').getContext('2d');
    
    // Count wins per model
    const wins = {
        'RLT': 0,
        'Random Forest': 0,
        'Gradient Boosting': 0,
        'ExtraTrees': 0,
        'Lasso/Ridge': 0
    };
    
    for (const dataset in PERFORMANCE_DATA) {
        const perf = PERFORMANCE_DATA[dataset];
        if (!perf.models) continue;
        
        let bestScore = -Infinity;
        let bestModel = null;
        
        for (const model in perf.models) {
            const modelPerf = perf.models[model];
            const score = perf.task_type === 'classification' 
                ? modelPerf.accuracy 
                : modelPerf.r2;
            
            if (score > bestScore) {
                bestScore = score;
                bestModel = model;
            }
        }
        
        if (bestModel) {
            // Map Ridge/Lasso to combined category
            if (bestModel === 'Logistic Ridge' || bestModel === 'Lasso') {
                wins['Lasso/Ridge']++;
            } else {
                wins[bestModel]++;
            }
        }
    }
    
    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: Object.keys(wins),
            datasets: [{
                data: Object.values(wins),
                backgroundColor: [
                    'rgba(91, 33, 182, 0.8)',
                    'rgba(5, 150, 105, 0.8)',
                    'rgba(6, 182, 212, 0.8)',
                    'rgba(239, 68, 68, 0.8)',
                    'rgba(245, 158, 11, 0.8)'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Number of Datasets Where Each Model Performed Best',
                    font: { size: 16, weight: 'bold' }
                },
                legend: {
                    position: 'bottom'
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ${context.parsed} wins (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

// ========== NEW VISUALIZATIONS ==========

async function loadROCCurves() {
    try {
        const response = await fetch('/api/roc-curves');
        const rocData = await response.json();
        
        const container = document.getElementById('roc-curves-container');
        container.innerHTML = '';
        
        for (const dataset in rocData) {
            const card = document.createElement('div');
            card.className = 'roc-card';
            
            const title = document.createElement('h3');
            title.textContent = dataset.replace(/_/g, ' ').toUpperCase();
            card.appendChild(title);
            
            const canvas = document.createElement('canvas');
            canvas.id = `roc-${dataset}`;
            card.appendChild(canvas);
            container.appendChild(card);
            
            const ctx = canvas.getContext('2d');
            const datasets = [];
            const colors = {
                'RLT': 'rgba(91, 33, 182, 0.8)',
                'Random Forest': 'rgba(5, 150, 105, 0.8)',
                'Gradient Boosting': 'rgba(6, 182, 212, 0.8)'
            };
            
            for (const model in rocData[dataset]) {
                const data = rocData[dataset][model];
                datasets.push({
                    label: `${model} (AUC=${data.auc.toFixed(3)})`,
                    data: data.fpr.map((fpr, i) => ({ x: fpr, y: data.tpr[i] })),
                    borderColor: colors[model] || 'rgba(0, 0, 0, 0.5)',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.1
                });
            }
            
            datasets.push({
                label: 'Random (AUC=0.5)',
                data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                borderColor: 'rgba(0, 0, 0, 0.3)',
                backgroundColor: 'transparent',
                borderWidth: 1,
                borderDash: [5, 5],
                pointRadius: 0
            });
            
            new Chart(ctx, {
                type: 'line',
                data: { datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    aspectRatio: 1.2,
                    plugins: {
                        legend: { position: 'bottom', labels: { font: { size: 10 } } }
                    },
                    scales: {
                        x: { type: 'linear', title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
                        y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error loading ROC curves:', error);
    }
}

function createPerformanceDistributions() {
    const classCtx = document.getElementById('classificationDistChart').getContext('2d');
    const regCtx = document.getElementById('regressionDistChart').getContext('2d');
    
    const classDatasets = {};
    const regDatasets = {};
    
    for (const dataset in PERFORMANCE_DATA) {
        if (PERFORMANCE_DATA[dataset].task_type === 'classification') {
            classDatasets[dataset] = PERFORMANCE_DATA[dataset];
        } else {
            regDatasets[dataset] = PERFORMANCE_DATA[dataset];
        }
    }
    
    // Classification chart
    const classModels = ['RLT', 'Random Forest', 'Gradient Boosting'];
    const classData = {
        labels: Object.keys(classDatasets).map(ds => ds.replace(/_/g, ' ')),
        datasets: classModels.map((model, idx) => ({
            label: model,
            data: Object.keys(classDatasets).map(ds => {
                const perf = classDatasets[ds].models && classDatasets[ds].models[model];
                return perf ? perf.accuracy * 100 : null;
            }),
            backgroundColor: idx === 0 ? 'rgba(91, 33, 182, 0.6)' : idx === 1 ? 'rgba(5, 150, 105, 0.6)' : 'rgba(6, 182, 212, 0.6)'
        }))
    };
    
    new Chart(classCtx, {
        type: 'bar',
        data: classData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Classification Accuracy (%)', font: { size: 14, weight: 'bold' } }
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'Accuracy (%)' } }
            }
        }
    });
    
    // Regression chart
    const regModels = ['RLT', 'Random Forest', 'Lasso'];
    const regData = {
        labels: Object.keys(regDatasets).map(ds => ds.replace(/_/g, ' ')),
        datasets: regModels.map((model, idx) => ({
            label: model,
            data: Object.keys(regDatasets).map(ds => {
                const perf = regDatasets[ds].models && regDatasets[ds].models[model];
                return perf ? perf.r2 * 100 : null;
            }),
            backgroundColor: idx === 0 ? 'rgba(91, 33, 182, 0.6)' : idx === 1 ? 'rgba(5, 150, 105, 0.6)' : 'rgba(245, 158, 11, 0.6)'
        }))
    };
    
    new Chart(regCtx, {
        type: 'bar',
        data: regData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: 'Regression R² Score (%)', font: { size: 14, weight: 'bold' } }
            },
            scales: {
                y: { beginAtZero: true, max: 100, title: { display: true, text: 'R² Score (%)' } }
            }
        }
    });
}

function createPerformanceHeatmap() {
    const ctx = document.getElementById('performanceHeatmap').getContext('2d');
    const datasets = Object.keys(PERFORMANCE_DATA);
    const models = ['RLT', 'Random Forest', 'Gradient Boosting', 'ExtraTrees'];
    const heatmapData = datasets.flatMap((dataset, y) =>
        models.map((model, x) => {
            const perf = PERFORMANCE_DATA[dataset];
            if (perf.models && perf.models[model]) {
                const value = perf.task_type === 'classification' ?
                    perf.models[model].accuracy * 100 : perf.models[model].r2 * 100;
                return { x: x, y: y, v: value };
            }
            return { x: x, y: y, v: 0 };
        })
    );
    
    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Performance',
                data: heatmapData,
                backgroundColor(context) {
                    const value = context.raw.v;
                    const alpha = Math.max(0.2, value / 100);
                    return `rgba(91, 33, 182, ${alpha})`;
                },
                pointRadius: 20,
                pointHoverRadius: 22
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                title: { display: true, text: 'Performance Heatmap (Darker = Better)', font: { size: 14, weight: 'bold' } },
                tooltip: {
                    callbacks: {
                        label(context) {
                            return `${datasets[context.raw.y]} - ${models[context.raw.x]}: ${context.raw.v.toFixed(2)}%`;
                        }
                    }
                }
            },
            scales: {
                x: { type: 'linear', min: -0.5, max: models.length - 0.5, ticks: { stepSize: 1, callback: (val) => models[val] || '' } },
                y: { type: 'linear', min: -0.5, max: datasets.length - 0.5, ticks: { stepSize: 1, callback: (val) => datasets[val] || '' } }
            }
        }
    });
}
