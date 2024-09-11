// Populate column select dropdown
function populateColumnSelect() {
    const columnTypeSelect = document.getElementById('column-type-select');
    const columnSelectContainer = document.getElementById('column-select-container');
    const columnSelect = document.getElementById('column-select');
    const selectedType = columnTypeSelect.value;
    
    columnSelect.innerHTML = '';
    columnSelectContainer.style.display = selectedType ? 'block' : 'none';
    columnSelect.disabled = !selectedType;
    
    if (selectedType) {
        const columns = allData.column_types[selectedType.toLowerCase()];
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.text = column;
            columnSelect.add(option);
        });
    }
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

        testResults.innerHTML += `
            <h3>${column}</h3>
            <table>
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                </thead>
                <tbody>
                    ${Object.entries(columnStats.distribution_comparison).map(([key, value], index) => `
                        <tr>
                            <td>${metricLabels[index]}</td>
                            <td>${typeof value === 'number' ? value.toFixed(4) : value}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        plot.innerHTML += `<img src="${columnStats.plot}" alt="${column} Plot">`;
        
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

// Populate classification metrics table
function populateMetricsTable() {
    const metricsTable = document.getElementById('metrics-table');
    const tbody = metricsTable.querySelector('tbody');
    
    allData.classification_metrics.forEach(metric => {
        const row = document.createElement('tr');
        
        Object.entries(metric).forEach(([key, value]) => {
            const cell = document.createElement('td');
            cell.textContent = key === 'Model' ? value : value.toFixed(4);
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
    
    causalPlot.innerHTML = 'Causal plot not yet implemented';
    
    causalMetrics.innerHTML = `
        <p>Average Treatment Effect (ATE): ${causalComparison.ate.toFixed(4)}</p>
        <p>Average Treatment Effect on the Treated (ATT): ${causalComparison.att.toFixed(4)}</p>
    `;
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
}

init();