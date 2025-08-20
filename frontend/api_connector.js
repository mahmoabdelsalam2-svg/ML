document.addEventListener('DOMContentLoaded', () => {
    // --- Shared Elements & Logic ---
    const resultsArea = document.getElementById('results-area');
    const TABULAR_BASE_URL = 'http://127.0.0.1:8000/api/v1/tabular';
    const IMAGE_BASE_URL = 'http://127.0.0.1:8000/api/v1/image';

    function displayStatus(message, status) {
        resultsArea.classList.remove('hidden');
        let statusHtml = `<div class="card-modern text-center"><h2 class="text-2xl font-bold mb-4">${status.charAt(0).toUpperCase() + status.slice(1)}</h2>`;
        if (['pending', 'training', 'loading_data', 'preprocessing', 'evaluating'].includes(status)) {
            statusHtml += `<div class="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>`;
        }
        statusHtml += `<p class="text-gray-600">${message}</p></div>`;
        resultsArea.innerHTML = statusHtml;
        resultsArea.scrollIntoView({ behavior: 'smooth' });
    }

    // === TABULAR AUTOML LOGIC ===
    const tabularFileUpload = document.getElementById('tabular-file-upload');
    const tabularFileDropArea = document.getElementById('tabular-file-drop-area');
    const tabularFileNameDisplay = document.getElementById('tabular-file-name-display');
    const tabularProblemTypeSelect = document.getElementById('tabular-problem-type');
    const tabularModelSelect = document.getElementById('tabular-model-select');
    const tabularTargetColumnInput = document.getElementById('tabular-target-column');
    const tabularStartAnalysisBtn = document.getElementById('tabular-start-analysis-btn');
    let selectedTabularFile = null;

    const tabularModelOptions = {
        'Classification': ["Support Vector Machine (SVM)", "Random Forest", "Logistic Regression"],
        'Regression': ["Linear Regression", "SVR", "Random Forest Regressor"]
    };

    tabularFileDropArea.addEventListener('click', () => tabularFileUpload.click());
    tabularFileUpload.addEventListener('change', (e) => handleTabularFileSelect(e.target.files));
    tabularProblemTypeSelect.addEventListener('change', () => {
        const selectedType = tabularProblemTypeSelect.value;
        tabularModelSelect.innerHTML = '<option value="">Select Model</option>';
        if (tabularModelOptions[selectedType]) {
            tabularModelSelect.disabled = false;
            tabularModelOptions[selectedType].forEach(model => {
                const option = document.createElement('option');
                option.value = model;
                option.textContent = model;
                tabularModelSelect.appendChild(option);
            });
        } else {
            tabularModelSelect.disabled = true;
            tabularModelSelect.innerHTML = '<option value="">Select Problem Type First</option>';
        }
    });
    tabularStartAnalysisBtn.addEventListener('click', startTabularAnalysis);

    function handleTabularFileSelect(files) {
        if (files.length > 0) {
            selectedTabularFile = files[0];
            tabularFileNameDisplay.textContent = selectedTabularFile.name;
        }
    }

    async function startTabularAnalysis() {
        if (!selectedTabularFile || !tabularProblemTypeSelect.value || !tabularModelSelect.value || !tabularTargetColumnInput.value) {
            alert('Please fill all fields for tabular analysis.');
            return;
        }
        const formData = new FormData();
        formData.append('file', selectedTabularFile);
        formData.append('problem_type', tabularProblemTypeSelect.value);
        formData.append('model_name', tabularModelSelect.value);
        formData.append('target_column', tabularTargetColumnInput.value);
        displayStatus('Starting tabular analysis...', 'pending');

        try {
            // BUG FIX: Changed '/analyze' to '/analyze-file' to match the backend router.
            const res = await fetch(`${TABULAR_BASE_URL}/analyze-file`, { method: 'POST', body: formData });
            if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
            const data = await res.json();
            pollTabularStatus(data.job_id);
        } catch (error) { displayStatus(`Error: ${error.message}`, 'failed'); }
    }

    function pollTabularStatus(jobId) {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${TABULAR_BASE_URL}/status/${jobId}`);
                const data = await res.json();
                displayStatus(`Analysis Status: ${data.status}...`, data.status);
                if (data.status === 'complete' || data.status === 'failed') {
                    clearInterval(interval);
                    if (data.status === 'complete') {
                        displayTabularResults(data.results);
                    }
                }
            } catch (error) { clearInterval(interval); }
        }, 3000);
    }

    function displayTabularResults(results) {
        const metricName = Object.keys(results.metrics)[0];
        const metricValue = Object.values(results.metrics)[0].toFixed(4);
        resultsArea.innerHTML = `<div class="card-modern text-center"><h2 class="text-3xl font-bold text-gray-800">Analysis Complete!</h2><p class="text-xl mt-4">${metricName.replace('_', ' ').toUpperCase()}: <strong class="text-purple-600">${metricValue}</strong></p></div>`;
    }

    // === IMAGE CLASSIFICATION LOGIC ===
    const imagePredictionSection = document.getElementById('image-prediction-section');
    const zipUploadInput = document.getElementById('zip-upload');
    const zipDropArea = document.getElementById('zip-drop-area');
    const zipNameDisplay = document.getElementById('zip-name-display');
    const startImageTrainBtn = document.getElementById('start-image-train-btn');
    const imagePredictInput = document.getElementById('image-predict-upload');
    const imageDropArea = document.getElementById('image-drop-area');
    let imageModelId = null;
    let selectedZipFile = null;

    zipDropArea.addEventListener('click', () => zipUploadInput.click());
    zipUploadInput.addEventListener('change', (e) => handleZipSelect(e.target.files));
    startImageTrainBtn.addEventListener('click', uploadAndTrainImageModel);
    imageDropArea.addEventListener('click', () => imagePredictInput.click());
    imagePredictInput.addEventListener('change', (e) => handleImagePredictSelect(e.target.files));

    function handleZipSelect(files) {
        if (files.length > 0) {
            selectedZipFile = files[0];
            zipNameDisplay.textContent = selectedZipFile.name;
        }
    }

    function handleImagePredictSelect(files) {
        if (files.length > 0 && imageModelId) {
            predictSingleImage(files[0]);
        } else if (!imageModelId) { alert('Please train a model first.'); }
    }

    async function uploadAndTrainImageModel() {
        if (!selectedZipFile) { alert('Please select a .zip dataset first.'); return; }
        displayStatus('Uploading image dataset...', 'pending');
        const formData = new FormData();
        formData.append('file', selectedZipFile);

        try {
            const uploadRes = await fetch(`${IMAGE_BASE_URL}/upload-dataset`, { method: 'POST', body: formData });
            if (!uploadRes.ok) { const err = await uploadRes.json(); throw new Error(err.detail); }
            const uploadData = await uploadRes.json();

            displayStatus('Dataset uploaded. Starting training...', 'training');
            const trainRes = await fetch(`${IMAGE_BASE_URL}/train-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: uploadData.dataset_id, epochs: 10 })
            });
            if (!trainRes.ok) { const err = await trainRes.json(); throw new Error(err.detail); }
            const trainData = await trainRes.json();
            imageModelId = trainData.model_id;
            pollImageStatus();
        } catch (error) { displayStatus(`Error: ${error.message}`, 'failed'); }
    }

    function pollImageStatus() {
        const interval = setInterval(async () => {
            try {
                const res = await fetch(`${IMAGE_BASE_URL}/model-status/${imageModelId}`);
                const data = await res.json();
                displayStatus(`Training Status: ${data.status}...`, data.status);
                if (data.status === 'ready' || data.status === 'failed') {
                    clearInterval(interval);
                    if (data.status === 'ready') {
                        displayStatus('Training complete! You can now upload an image to predict.', 'complete');
                        imagePredictionSection.classList.remove('hidden');
                    }
                }
            } catch (error) { clearInterval(interval); }
        }, 3000);
    }

    async function predictSingleImage(imageFile) {
        displayStatus('Predicting image...', 'pending');
        const formData = new FormData();
        formData.append('file', imageFile);

        try {
            const res = await fetch(`${IMAGE_BASE_URL}/predict-image/${imageModelId}`, { method: 'POST', body: formData });
            if (!res.ok) { const err = await res.json(); throw new Error(err.detail); }
            const results = await res.json();
            displayImageResults(results);
        } catch (error) { displayStatus(`Error: ${error.message}`, 'failed'); }
    }

    function displayImageResults(results) {
        const confidence = (results.confidence * 100).toFixed(2);
        resultsArea.innerHTML = `<div class="card-modern text-center"><h2 class="text-3xl font-bold text-gray-800">Prediction Result</h2><p class="text-xl mt-4">Predicted Class: <strong class="text-purple-600">${results.predicted_class}</strong></p><p class="text-lg mt-2">Confidence: <strong class="text-purple-600">${confidence}%</strong></p></div>`;
    }
});