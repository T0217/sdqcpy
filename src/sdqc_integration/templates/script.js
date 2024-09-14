// Populate column select dropdown
function populateColumnSelect() {
    const columnTypeSelect = document.getElementById('column-type-select');
    const columnSelect = document.getElementById('column-select');
    const selectedType = columnTypeSelect.value;

    columnSelect.innerHTML = '';
    
    if (selectedType) {
        const columns = allData.column_types[selectedType.toLowerCase()];
        columns.forEach((column) => {
            const option = document.createElement('option');
            option.value = column;
            option.text = column;
            columnSelect.add(option);
        });
    }
    
    columnTypeSelect.dispatchEvent(new Event('change'));
    columnSelect.dispatchEvent(new Event('change'));
}

// Show statistics for selected columns
function showSelectedColumnStats() {
    const columnSelect = document.getElementById('column-select');
    const testResults = document.getElementById('test-results');
    const plot = document.getElementById('plot');
    const moreInfoTable = document.getElementById('more-info-table').querySelector('tbody');
    
    testResults.innerHTML = '';
    plot.innerHTML = '';
    moreInfoTable.innerHTML = '';
    
    for (const option of columnSelect.selectedOptions) {
        const column = option.value;
        const columnStats = allData.stats[column];
        let metricLabels;
        if (allData.column_types.categorical.includes(column)) {
            metricLabels = ['Jaccard Index', 'Chi-square Goodness of Fit'];
        } else {
            metricLabels = ['Wasserstein Distance', 'Hellinger Distance'];
        }

        const columnStatsTemplate = document.getElementById('column-stats-template');
        const columnStatsElement = columnStatsTemplate.content.cloneNode(true);
        
        columnStatsElement.querySelector('h3').textContent = column;
        const tbody = columnStatsElement.querySelector('tbody');
        
        Object.entries(columnStats.distribution_comparison).forEach(([key, value], index) => {
            const row = document.createElement('tr');
            const metricCell = document.createElement('td');
            const valueCell = document.createElement('td');
            
            metricCell.textContent = metricLabels[index];
            valueCell.textContent = typeof value === 'number' ? value.toFixed(3) : value;
            
            row.appendChild(metricCell);
            row.appendChild(valueCell);
            tbody.appendChild(row);
        });
        
        testResults.appendChild(columnStatsElement);
        
        const plotTemplate = document.getElementById('plot-template');
        const plotElement = plotTemplate.content.cloneNode(true);
        
        const img = plotElement.querySelector('img');
        img.src = columnStats.plot;
        img.alt = `${column} Plot`;
        
        plot.appendChild(plotElement);
        
        Object.entries(columnStats.raw_stats).forEach(([key, value]) => {
            const row = document.createElement('tr');
            
            const statCell = document.createElement('td');
            statCell.textContent = key;
            row.appendChild(statCell);
            
            const rawCell = document.createElement('td');
            rawCell.textContent = value;
            row.appendChild(rawCell);
            
            const synthCell = document.createElement('td');
            synthCell.textContent = columnStats.synth_stats[key];
            row.appendChild(synthCell);
            
            moreInfoTable.appendChild(row);
        });
    }
}

// Toggle more information
function toggleMoreInfo() {
    const moreInfo = document.getElementById('more-info');
    moreInfo.style.display = moreInfo.style.display === 'none' ? 'block' : 'none';
}

// Populate correlation selects
function populateCorrelationSelects() {
    const column1Select = document.getElementById('correlation-column1-select');
    const column2Select = document.getElementById('correlation-column2-select');
    
    const columns = Object.values(allData.column_types.categorical).concat(Object.values(allData.column_types.numerical));
    columns.forEach(column => {
        const option1 = document.createElement('option');
        option1.value = column;
        option1.textContent = column;
        column1Select.appendChild(option1);
    });

    function updateColumn2Options() {
        column2Select.innerHTML = '';
        columns.forEach(column => {
            if (column !== column1Select.value) {
                const option2 = document.createElement('option');
                option2.value = column;
                option2.textContent = column;
                column2Select.appendChild(option2);
            }
        });
    }

    column1Select.addEventListener('change', updateColumn2Options);
    updateColumn2Options();

    column1Select.addEventListener('change', updateCorrelationPlot);
    column2Select.addEventListener('change', updateCorrelationPlot);

    // Initialize correlation plot
    updateCorrelationPlot();
    
    // Add help text
    const correlationPlotContainer = document.getElementById('correlation-column-select');
    const helpText = document.createElement('p');
    helpText.textContent = 'Please select two columns to view their correlation.';
    helpText.style.fontStyle = 'italic';
    helpText.style.color = '#666';
    correlationPlotContainer.insertBefore(helpText, correlationPlotContainer.firstChild);
}

// Update correlation plot
function updateCorrelationPlot() {
    const column1Select = document.getElementById('correlation-column1-select');
    const column2Select = document.getElementById('correlation-column2-select');
    const correlationPlot = document.getElementById('correlation-image');
    
    const selectedColumns = `${column1Select.value} vs ${column2Select.value}`;
    correlationPlot.src = allData.correlation_plots[selectedColumns];
    correlationPlot.alt = `${selectedColumns} Correlation Plot`;
}

// Populate classification metrics table
function populateMetricsTable() {
    const metricsTable = document.getElementById('metrics-table');
    const tbody = metricsTable.querySelector('tbody');
    const modelMapping = {
        'svm': 'SVM',
        'rf': 'RandomForest',
        'xgb': 'XGBoost',
        'lgbm': 'LightGBM'
    };
    
    allData.classification_metrics.forEach(metric => {
        const row = document.createElement('tr');
        
        Object.entries(metric).forEach(([key, value]) => {
            const cell = document.createElement('td');
            cell.textContent = key === 'Model' ? modelMapping[value] : value.toFixed(4);
            row.appendChild(cell);
        });
        
        tbody.appendChild(row);
    });
}

// Show feature importance table or visualization
function showFeatureImportance() {
    const featureImportance = allData.feature_importance;
    const fiDisplaySelect = document.getElementById('fi-display-select');
    const fiTable = document.getElementById('fi-table');
    const fiVisualization = document.getElementById('fi-visualization');
    const fiImage = document.getElementById('fi-image');
    
    if (fiDisplaySelect.value === 'table') {
        fiTable.style.display = 'table';
        fiVisualization.style.display = 'none';
        
        const tbody = fiTable.querySelector('tbody');
        tbody.innerHTML = '';
        
        const importantFeatures = featureImportance[0].filter(feature => feature.importance >= 0.01);
        const otherFeatures = featureImportance[0].filter(feature => feature.importance < 0.01);
        
        importantFeatures.forEach(feature => {
            const row = document.createElement('tr');
            
            const featureCell = document.createElement('td');
            featureCell.textContent = feature.feature;
            row.appendChild(featureCell);
            
            const importanceCell = document.createElement('td');
            importanceCell.textContent = feature.importance.toFixed(4);
            row.appendChild(importanceCell);
            
            tbody.appendChild(row);
        });
        
        const otherFeaturesTbody = document.getElementById('other-features-tbody');
        otherFeaturesTbody.innerHTML = otherFeatures.map(feature => `
            <tr>
                <td>${feature.feature}</td>
                <td>${feature.importance.toFixed(4)}</td>
            </tr>
        `).join('');
    } else {
        fiTable.style.display = 'none';
        fiVisualization.style.display = 'block';
        
        fiImage.src = featureImportance[1];
        fiImage.alt = 'Feature Importance Visualization';
    }
}

// Show causal comparison results
function showCausalComparison() {
    const causalComparison = allData.causal_comparison;
    const causalPlot = document.getElementById('causal-plot');
    const causalMetrics = document.getElementById('causal-metrics');
    const tbody = causalMetrics.querySelector('tbody');
    
    causalPlot.src = causalComparison['Adjacency_matrices'];
    
    tbody.innerHTML = Object.entries(causalComparison.causal_metrics)
        .map(([key, value]) => `
            <tr>
                <td>${key}</td>
                <td>${value.toFixed(value.toString().split('.')[1]?.length > 3 ? 3 : value.toString().split('.')[1]?.length || 0)}</td>
            </tr>
        `).join('');
}

// Update summary section
function updateSummary() {
    updateDuplicateSummary();
    updateDistributionSummary();
    updateClassificationSummary();
    updateCausalSummary();
}

// Update duplicate summary
function updateDuplicateSummary() {
    const duplicateElement = document.getElementById('duplicate-summary');
    const duplicateSituation = allData.duplicate_situation

    let summaryText = ``;
    let problemColumns = allData.column_types.problem || [];
    if (duplicateSituation){
        summaryText += ` There are <strong>duplicate data</strong> between the synthetic data and the original data.`;
    } else {
        summaryText += ` There are <strong>no duplicate data</strong> between the synthetic data and the original data.`;
    }
     
    if (problemColumns.length > 0) {
        summaryText += `
        <br>The following columns have issues: <strong style="color: red;">${problemColumns.join(', ')}</strong>.
        <br>This may be due to the presence of string variables in the data or insufficient data volume.
        `;
    }
    
    duplicateElement.innerHTML = summaryText;
}

// Update distribution summary
function updateDistributionSummary() {
    const distributionElement = document.getElementById('distribution-summary');
    let largeDiscrepancyColumns = [];

    for (const column in allData.stats) {
        const stats = allData.stats[column];
        if (allData.column_types.categorical.includes(column)) {
            if (stats.distribution_comparison['chi_square_p_value'] < 0.8) {
                largeDiscrepancyColumns.push(column);
            }
        } else {
            const wasserstein = stats.distribution_comparison['wasserstein_distance'];
            const hellinger = stats.distribution_comparison['hellinger_distance'];
            if (wasserstein < 10 || hellinger < 10) {
                largeDiscrepancyColumns.push(column);
            }
        }
    }

    let summaryText = 'Columns with large distribution discrepancies: ';
    if (largeDiscrepancyColumns.length > 0) {
        summaryText += `<strong style="color: red;">${largeDiscrepancyColumns.join(', ')}</strong>.`;
    } else {
        summaryText += 'None.';
    }

    distributionElement.innerHTML = summaryText;
}

// Update classification summary
function updateClassificationSummary() {
    const classificationElement = document.getElementById('classification-summary');
    const metrics = allData.classification_metrics;
    
    let maxAUC = 0;
    let maxAUCModel = '';
    let allBelowThreshold = true;

    metrics.forEach(metric => {
        if (metric.AUC > maxAUC) {
            maxAUC = metric.AUC;
            const modelMapping = {
                'svm': 'SVM',
                'rf': 'RandomForest',
                'xgb': 'XGBoost',
                'lgbm': 'LightGBM'
            };
            maxAUCModel = modelMapping[metric.Model];
        }
        if (metric.AUC >= 0.5) {
            allBelowThreshold = false;
        }
    });

    let summaryText = '';
    if (allBelowThreshold) {
        summaryText = 'The difference between synthetic data and original data is small.';
    } else {
        summaryText = `The model with the highest AUC is <strong>${maxAUCModel}</strong>, with an AUC value of <strong>${maxAUC.toFixed(3)}</strong>.`;
    }

    // Get top 5 feature importances
    const topFeatures = allData.feature_importance[0]
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 5);

    summaryText += '<p style="margin-top: 10px;"><strong>Top 5 feature importances:</strong></p><ol>';
    topFeatures.forEach((feature, index) => {
        summaryText += `<li>${feature.feature} : ${feature.importance.toFixed(3)}</li>`;
    });
    summaryText += '</ol>';

    classificationElement.innerHTML = summaryText;
}

// Update causal summary
function updateCausalSummary() {
    const causalElement = document.getElementById('causal-summary');
    const gscore = allData.causal_comparison.causal_metrics.gscore;

    let summaryText = `The difference in causal identification is `;
    if (gscore > 0.6) {
        summaryText += '<strong style="color: green;">small</strong>';
    } else {
        summaryText += '<strong style="color: red;">large</strong>';
    }
    summaryText += ` (G-Score: ${gscore.toFixed(3)}).`;

    causalElement.innerHTML = summaryText;
}

// Initialize page
function init() {
    const columnTypeSelect = document.getElementById('column-type-select');
    columnTypeSelect.addEventListener('change', populateColumnSelect);
    populateColumnSelect();
    
    const columnSelect = document.getElementById('column-select');
    columnSelect.addEventListener('change', showSelectedColumnStats);
    
    const moreInfoBtn = document.getElementById('more-info-btn');
    moreInfoBtn.addEventListener('click', toggleMoreInfo);
    
    const fiDisplaySelect = document.getElementById('fi-display-select');
    fiDisplaySelect.addEventListener('change', showFeatureImportance);
    
    populateMetricsTable();
    showSelectedColumnStats();
    showFeatureImportance();
    showCausalComparison();
    
    populateCorrelationSelects();
}

// Run initialization when DOM is fully loaded
document.addEventListener("DOMContentLoaded", () => {
    init();
    updateSummary();
});