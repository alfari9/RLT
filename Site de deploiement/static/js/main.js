// État global de l'application
let currentDataset = DATASET_NAME || null;
let datasetInfo = null;

// Initialisation au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    if (currentDataset) {
        // Mode prédiction (predict.html)
        loadDatasetInfo();
        setupEventListeners();
    } else {
        // Mode liste (index.html old)
        loadDatasets();
        setupEventListeners();
    }
});

// Charger les informations du dataset (mode predict.html)
async function loadDatasetInfo() {
    try {
        const response = await fetch(`/api/dataset/${currentDataset}/info`);
        datasetInfo = await response.json();
        
        // Update page title and description
        document.getElementById('dataset-description').textContent = datasetInfo.metadata.description || '';
        
        // Display dataset info card
        displayDatasetInfoCard(datasetInfo);
        
        // Display features form
        displayFeaturesForm(datasetInfo);
        
    } catch (error) {
        console.error('Error loading dataset info:', error);
        alert('Error loading dataset information');
    }
}

// Display dataset info card
function displayDatasetInfoCard(info) {
    const infoCard = document.getElementById('dataset-info-card');
    const taskBadge = info.metadata.task_type === 'classification' 
        ? '<span class="dataset-badge dataset-badge-classification">Classification</span>'
        : '<span class="dataset-badge dataset-badge-regression">Regression</span>';
    
    infoCard.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: start;">
            <div>
                <h3 style="margin-bottom: 1rem;">Dataset Information</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem;">
                    <div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Task Type</div>
                        <div style="margin-top: 0.25rem;">${taskBadge}</div>
                    </div>
                    <div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Samples</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">${info.metadata.n_samples}</div>
                    </div>
                    <div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Features</div>
                        <div style="font-size: 1.5rem; font-weight: 700; color: var(--primary-color);">${info.metadata.n_features}</div>
                    </div>
                    <div>
                        <div style="color: var(--text-secondary); font-size: 0.9rem;">Target Range</div>
                        <div style="font-size: 1.25rem; font-weight: 600;">${info.metadata.target_info.min.toFixed(2)} - ${info.metadata.target_info.max.toFixed(2)}</div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Charger la liste des datasets (mode ancien)
async function loadDatasets() {
    try {
        const response = await fetch('/api/datasets');
        const datasets = await response.json();
        
        const container = document.getElementById('datasets-list');
        if (!container) return;
        container.innerHTML = '';
        
        datasets.forEach(dataset => {
            const card = createDatasetCard(dataset);
            container.appendChild(card);
        });
    } catch (error) {
        console.error('Erreur lors du chargement des datasets:', error);
        alert('Erreur lors du chargement des datasets');
    }
}

// Créer une carte de dataset
function createDatasetCard(dataset) {
    const card = document.createElement('div');
    card.className = 'dataset-card';
    card.dataset.folderName = dataset.folder_name;
    
    const badgeClass = dataset.task_type === 'classification' ? 'badge-classification' : 'badge-regression';
    
    card.innerHTML = `
        <span class="dataset-badge ${badgeClass}">${dataset.task_type.toUpperCase()}</span>
        <h3>${dataset.display_name}</h3>
        <div class="dataset-stats">
            <span>📊 ${dataset.n_samples} échantillons</span>
            <span>🔢 ${dataset.n_features} features</span>
        </div>
    `;
    
    card.addEventListener('click', () => selectDataset(dataset.folder_name, card));
    
    return card;
}

// Sélectionner un dataset
async function selectDataset(folderName, cardElement) {
    // Retirer la sélection précédente
    document.querySelectorAll('.dataset-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    // Ajouter la sélection à la carte cliquée
    cardElement.classList.add('selected');
    
    currentDataset = folderName;
    
    // Charger les infos du dataset
    try {
        const response = await fetch(`/api/dataset/${folderName}/info`);
        datasetInfo = await response.json();
        
        displayDatasetInfo(datasetInfo);
        displayFeaturesForm(datasetInfo);
        
    } catch (error) {
        console.error('Erreur lors du chargement des infos:', error);
        alert('Erreur lors du chargement des informations du dataset');
    }
}

// Afficher les informations du dataset (legacy - for old index.html)
function displayDatasetInfo(info) {
    const infoSection = document.getElementById('dataset-info');
    const detailsContainer = document.getElementById('dataset-details');
    
    // Check if we're on old page
    if (infoSection && detailsContainer) {
        infoSection.classList.remove('hidden');
        
        detailsContainer.innerHTML = `
            <div class="info-item">
                <strong>Type de tâche</strong>
                <span>${info.metadata.task_type}</span>
            </div>
            <div class="info-item">
                <strong>Nombre d'échantillons</strong>
                <span>${info.metadata.n_samples}</span>
            </div>
            <div class="info-item">
                <strong>Nombre de features</strong>
                <span>${info.metadata.n_features}</span>
            </div>
            <div class="info-item">
                <strong>Plage cible</strong>
                <span>${info.metadata.target_info.min.toFixed(2)} - ${info.metadata.target_info.max.toFixed(2)}</span>
            </div>
        `;
        
        // Scroll vers le formulaire
        const featuresSection = document.getElementById('features-section');
        if (featuresSection) {
            featuresSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
}

// Afficher le formulaire de saisie des features
function displayFeaturesForm(info) {
    const inputsContainer = document.getElementById('features-inputs');
    
    if (!inputsContainer) {
        console.error('features-inputs container not found');
        return;
    }
    
    inputsContainer.innerHTML = '';
    
    // Get real feature ranges (not scaled)
    const featureRanges = info.feature_ranges || {};
    
    info.metadata.feature_names.forEach((featureName, index) => {
        // Use real ranges from feature_ranges.json, fallback to metadata stats
        const realRange = featureRanges[featureName.toLowerCase()];
        const minVal = realRange ? realRange.min : info.metadata.feature_stats[index].min;
        const maxVal = realRange ? realRange.max : info.metadata.feature_stats[index].max;
        const meanVal = realRange ? realRange.mean : info.metadata.feature_stats[index].mean;
        
        const inputDiv = document.createElement('div');
        inputDiv.className = 'feature-input-group';
        
        inputDiv.innerHTML = `
            <label for="feature-${index}" title="Min: ${minVal.toFixed(2)}, Max: ${maxVal.toFixed(2)}, Mean: ${meanVal.toFixed(2)}">
                ${featureName}
            </label>
            <input 
                type="number" 
                id="feature-${index}" 
                name="${featureName}" 
                step="any"
                placeholder="Min: ${minVal.toFixed(2)}, Max: ${maxVal.toFixed(2)}"
                required
            >
        `;
        
        inputsContainer.appendChild(inputDiv);
    });
}

// Remplissage aléatoire intelligent
async function randomFill() {
    if (!currentDataset) return;
    
    try {
        const response = await fetch(`/api/dataset/${currentDataset}/random-fill`, {
            method: 'POST'
        });
        const randomValues = await response.json();
        
        // Remplir le formulaire avec les valeurs aléatoires
        Object.entries(randomValues).forEach(([featureName, value]) => {
            const input = document.querySelector(`input[name="${featureName}"]`);
            if (input) {
                input.value = value.toFixed(4);
            }
        });
        
    } catch (error) {
        console.error('Erreur lors du remplissage aléatoire:', error);
        alert('Erreur lors du remplissage aléatoire');
    }
}

// Effacer le formulaire
function clearForm() {
    document.getElementById('features-form').reset();
}

// Soumettre les prédictions
async function submitPredictions(event) {
    event.preventDefault();
    
    if (!currentDataset) {
        alert('Please select a dataset first');
        return;
    }
    
    // Afficher le loading
    const resultsSection = document.getElementById('results-section');
    const loading = document.getElementById('loading');
    const resultsContainer = document.getElementById('results-grid') || document.getElementById('results-content');
    
    if (!resultsSection || !loading) {
        console.error('Results section elements not found');
        return;
    }
    
    resultsSection.classList.remove('hidden');
    loading.classList.remove('hidden');
    if (resultsContainer) resultsContainer.innerHTML = '';
    
    // Scroll vers les résultats
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Récupérer les valeurs du formulaire
    const formData = new FormData(event.target);
    const features = {};
    
    for (let [key, value] of formData.entries()) {
        features[key] = parseFloat(value);
    }
    
    console.log('Submitting prediction for:', currentDataset, features);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_name: currentDataset,
                features: features
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const results = await response.json();
        console.log('Received results:', results);
        
        // Masquer le loading
        loading.classList.add('hidden');
        
        // Afficher les résultats
        displayResults(results);
        
    } catch (error) {
        console.error('Error during prediction:', error);
        loading.classList.add('hidden');
        if (resultsContainer) {
            resultsContainer.innerHTML = `
                <div style="padding: 2rem; text-align: center; color: var(--danger-color);">
                    <h3>❌ Error</h3>
                    <p>${error.message}</p>
                </div>
            `;
        }
        alert('Error during prediction: ' + error.message);
    }
}

// Afficher les résultats
function displayResults(results) {
    const container = document.getElementById('results-grid') || document.getElementById('results-content');
    if (!container) {
        console.error('Results container not found');
        return;
    }
    container.innerHTML = '';
    
    // Trouver le meilleur modèle (plus proche de la moyenne pour régression, plus haute accuracy pour classification)
    let bestModel = null;
    let bestValue = results.task_type === 'classification' ? -Infinity : null;
    
    const predictions = Object.entries(results.predictions);
    
    // Pour la classification, trouver le modèle avec la meilleure accuracy historique
    if (results.task_type === 'classification') {
        predictions.forEach(([modelName, data]) => {
            if (!data.error) {
                const accuracy = data.historical_performance?.accuracy || 0;
                if (accuracy > bestValue) {
                    bestValue = accuracy;
                    bestModel = modelName;
                }
            }
        });
    } else {
        // Pour la régression, calculer la moyenne des prédictions
        const validPredictions = predictions.filter(([_, data]) => !data.error).map(([_, data]) => data.prediction);
        const avgPrediction = validPredictions.reduce((a, b) => a + b, 0) / validPredictions.length;
        
        // Trouver le modèle le plus proche de la moyenne
        let minDiff = Infinity;
        predictions.forEach(([modelName, data]) => {
            if (!data.error) {
                const diff = Math.abs(data.prediction - avgPrediction);
                if (diff < minDiff) {
                    minDiff = diff;
                    bestModel = modelName;
                }
            }
        });
    }
    
    // Créer les cartes de résultats
    predictions.forEach(([modelName, data]) => {
        let card;
        if (data.error) {
            // Show error card instead of hiding
            card = createErrorCard(modelName, data.error);
        } else {
            card = createResultCard(modelName, data, results.task_type, modelName === bestModel);
        }
        container.appendChild(card);
    });
    
    // Ajouter un graphique de comparaison
    const comparisonChart = createComparisonChart(predictions, results.task_type);
    container.appendChild(comparisonChart);
}

// Créer une carte d'erreur
function createErrorCard(modelName, errorMessage) {
    const card = document.createElement('div');
    card.className = 'result-card error-card';
    
    card.innerHTML = `
        <div class="result-header">
            <h3>${modelName}</h3>
            <span class="error-badge">❌ Erreur</span>
        </div>
        <div class="error-message">
            <strong>Erreur:</strong>
            <p>${errorMessage}</p>
        </div>
    `;
    
    return card;
}

// Créer une carte de résultat
function createResultCard(modelName, data, taskType, isBest) {
    const card = document.createElement('div');
    card.className = `result-card ${isBest ? 'best-model' : ''}`;
    
    const predictionLabel = taskType === 'classification' ? 'Classe prédite' : 'Valeur prédite';
    const predictionValue = taskType === 'classification' ? 
        (data.prediction === 1 ? 'Classe 1' : 'Classe 0') : 
        data.prediction.toFixed(4);
    
    let metricsHTML = '';
    if (data.historical_performance) {
        const perf = data.historical_performance;
        
        if (taskType === 'classification') {
            metricsHTML = `
                <div class="performance-metrics">
                    <div class="metric-item">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">${(perf.accuracy * 100).toFixed(2)}%</div>
                    </div>
                </div>
            `;
        } else {
            metricsHTML = `
                <div class="performance-metrics">
                    <div class="metric-item">
                        <div class="metric-label">MSE</div>
                        <div class="metric-value">${perf.mse?.toFixed(4) || 'N/A'}</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-label">R²</div>
                        <div class="metric-value">${perf.r2?.toFixed(4) || 'N/A'}</div>
                    </div>
                </div>
            `;
        }
    }
    
    card.innerHTML = `
        <div class="result-header">
            <h3>${modelName}</h3>
            ${isBest ? '<span class="best-badge">🏆 Meilleur Modèle</span>' : ''}
        </div>
        <div>
            <strong>${predictionLabel}:</strong>
            <div class="prediction-value">${predictionValue}</div>
        </div>
        ${metricsHTML}
    `;
    
    return card;
}

// Créer un graphique de comparaison
function createComparisonChart(predictions, taskType) {
    const chart = document.createElement('div');
    chart.className = 'comparison-chart';
    chart.innerHTML = '<h3>📊 Comparaison des Prédictions</h3>';
    
    // Filtrer les prédictions valides
    const validPredictions = predictions.filter(([_, data]) => !data.error);
    
    if (taskType === 'regression') {
        // Pour la régression, afficher les valeurs prédites
        const values = validPredictions.map(([_, data]) => data.prediction);
        const maxValue = Math.max(...values);
        
        validPredictions.forEach(([modelName, data]) => {
            const percentage = (data.prediction / maxValue) * 100;
            
            const barDiv = document.createElement('div');
            barDiv.className = 'chart-bar';
            barDiv.innerHTML = `
                <div class="chart-label">${modelName}</div>
                <div class="chart-bar-container">
                    <div class="chart-bar-fill" style="width: ${percentage}%">
                        ${data.prediction.toFixed(4)}
                    </div>
                </div>
            `;
            chart.appendChild(barDiv);
        });
    } else {
        // Pour la classification, afficher les accuracies historiques
        validPredictions.forEach(([modelName, data]) => {
            const accuracy = (data.historical_performance?.accuracy || 0) * 100;
            
            const barDiv = document.createElement('div');
            barDiv.className = 'chart-bar';
            barDiv.innerHTML = `
                <div class="chart-label">${modelName}</div>
                <div class="chart-bar-container">
                    <div class="chart-bar-fill" style="width: ${accuracy}%">
                        ${accuracy.toFixed(2)}%
                    </div>
                </div>
            `;
            chart.appendChild(barDiv);
        });
    }
    
    return chart;
}

// Configuration des event listeners
function setupEventListeners() {
    const randomBtn = document.getElementById('random-fill-btn');
    const clearBtn = document.getElementById('clear-form-btn');
    const form = document.getElementById('features-form');
    
    if (randomBtn) randomBtn.addEventListener('click', randomFill);
    if (clearBtn) clearBtn.addEventListener('click', clearForm);
    if (form) form.addEventListener('submit', submitPredictions);
}
