// Configuration Data (You should replace these with your real data points)
const dataConfig = {
    'DAUC': {
        label: 'Deletion Curve (mIoU)',
        // Example data: mIoU drops as we delete pixels
        data: [0.85, 0.70, 0.55, 0.40, 0.25, 0.15, 0.10, 0.08, 0.05, 0.02, 0.01], 
        color: 'rgb(255, 99, 132)', // Red
        folder: 'dauc', // Folder name for images
        insight: "removing top relevant pixels causes a sharp drop in segmentation quality."
    },
    'IAUC': {
        label: 'Insertion Curve (mIoU)',
        // Example data: mIoU rises as we insert pixels
        data: [0.05, 0.15, 0.35, 0.55, 0.70, 0.78, 0.82, 0.84, 0.85, 0.85, 0.85],
        color: 'rgb(75, 192, 192)', // Teal/Green
        folder: 'iauc',
        insight: "adding just the top 20% of relevant pixels recovers most of the performance."
    }
};

let currentMetric = 'DAUC';
let causalChart = null;

// Initialize the Chart
function initChart() {
    const ctx = document.getElementById('causalChart').getContext('2d');
    const steps = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]; // X-Axis Labels

    causalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: steps,
            datasets: [{
                label: dataConfig['DAUC'].label,
                data: dataConfig['DAUC'].data,
                borderColor: dataConfig['DAUC'].color,
                tension: 0.3,
                pointRadius: 5, // Default point size
                pointHoverRadius: 7,
                pointBackgroundColor: dataConfig['DAUC'].color
            }]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                tooltip: {
                    enabled: true
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    title: {
                        display: true,
                        text: 'mIoU'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '% of Pixels Perturbed'
                    }
                }
            }
        }
    });
}

// Handle Slider Updates
function updateVisualization(value) {
    // 1. Update Label
    document.getElementById('step-label').innerText = value + "%";

    // 2. Update Chart Highlight
    // We find the index corresponding to the slider value (0=0, 10=1, etc.)
    const index = value / 10; 
    
    // Reset all point styles
    const pointRadii = new Array(11).fill(5);
    const pointColors = new Array(11).fill(dataConfig[currentMetric].color);

    // Highlight the selected point
    pointRadii[index] = 12; // Make it big
    pointColors[index] = 'rgb(50, 50, 50)'; // Make it dark grey/black

    causalChart.data.datasets[0].pointRadius = pointRadii;
    causalChart.data.datasets[0].pointBackgroundColor = pointColors;
    causalChart.update('none'); // 'none' mode prevents full re-animation

    // 3. Update Images
    // Assumes files are named like: ./static/images/dauc/support_20.png
    const folder = dataConfig[currentMetric].folder;
    document.getElementById('dynamic-support-img').src = `./static/images/${folder}/support_${value}.png`;
    document.getElementById('dynamic-query-img').src = `./static/images/${folder}/query_${value}.png`;
}

// Handle Tab Switching
function switchMetric(metric) {
    currentMetric = metric;

    // Update Tabs UI
    document.getElementById('tab-dauc').classList.remove('is-active');
    document.getElementById('tab-iauc').classList.remove('is-active');
    document.getElementById(`tab-${metric.toLowerCase()}`).classList.add('is-active');

    // Update Chart Data
    causalChart.data.datasets[0].label = dataConfig[metric].label;
    causalChart.data.datasets[0].data = dataConfig[metric].data;
    causalChart.data.datasets[0].borderColor = dataConfig[metric].color;
    causalChart.data.datasets[0].pointBackgroundColor = dataConfig[metric].color;
    
    // Update Insight Text
    document.getElementById('metric-insight').innerText = dataConfig[metric].insight;

    // Reset Slider and View
    document.getElementById('step-slider').value = 0;
    updateVisualization(0);
}

// Initialize on Load
document.addEventListener('DOMContentLoaded', function () {
    initChart();
    updateVisualization(0); // Initial highlight
});