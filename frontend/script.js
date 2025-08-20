document.addEventListener('DOMContentLoaded', function() {
    // --- Configuration ---
    const API_BASE_URL = 'http://127.0.0.1:8000/api/v1';

    // --- Model options for dynamic dropdowns with categories ---
    const modelOptions = {
        'Classification': ["Random Forest", "Logistic Regression", "Decision Tree", "Support Vector Machine (SVM)"],
        'Regression': ["Linear Regression", "Support Vector Regressor (SVR)", "Random Forest Regressor"]
    };

    // --- Global State ---
    let activeJobId = null;
    let activeModelId = null;
    let jobPollingInterval = null;

    // --- Element Selectors ---
    const dynamicContentArea = document.getElementById('dynamic-content-area');
    const chatbotBubble = document.getElementById('chatbot-bubble');
    const chatbotWindow = document.getElementById('chatbot-window');
    const closeChatBtn = document.getElementById('close-chat-btn');
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const chatSendBtn = document.getElementById('chat-send-btn');
    const menuToggle = document.getElementById("menu-toggle");
    const menuDropdown = document.getElementById("menu-dropdown");

    // --- INITIALIZATION ---
    setupEventListeners();

    // --- EVENT LISTENERS SETUP ---
    function setupEventListeners() {
        menuToggle?.addEventListener("click", toggleMenu);
        document.addEventListener("click", closeMenuOnClickOutside);
        document.querySelectorAll('#menu-dropdown a[data-target]').forEach(a => a.addEventListener("click", handleSmoothScroll));

        document.getElementById('upload-start-analysis-btn').addEventListener('click', handleTabularFileUpload);
        document.getElementById('manual-start-analysis-btn').addEventListener('click', handleTabularUrlManual);
        document.getElementById('automl-start-analysis-btn').addEventListener('click', handleTabularUrlAutoML);

        document.getElementById('image-train-btn').addEventListener('click', handleImageTrain);
        document.getElementById('image-predict-btn').addEventListener('click', handleImagePredict);

        document.querySelectorAll('.file-drop-area').forEach(area => {
            area.addEventListener('click', () => document.getElementById(area.dataset.inputId).click());
            area.addEventListener('dragover', e => { e.preventDefault(); area.classList.add('border-purple-500', 'bg-lavender'); });
            area.addEventListener('dragleave', () => area.classList.remove('border-purple-500', 'bg-lavender'));
            area.addEventListener('drop', handleFileDrop);
        });

        document.querySelectorAll('input[type="file"]').forEach(input => input.addEventListener('change', handleFileChange));

        chatbotBubble.addEventListener('click', toggleChatbotWindow);
        closeChatBtn.addEventListener('click', () => chatbotWindow.classList.remove('open'));
        chatSendBtn.addEventListener('click', sendChatMessage);
        chatInput.addEventListener('keypress', e => { if (e.key === 'Enter') sendChatMessage(); });
        document.addEventListener('click', closeChatbotOnClickOutside);

        document.querySelectorAll('.custom-select-wrapper').forEach(setupCustomSelect);
    }

    // --- CUSTOM DROPDOWN LOGIC ---
    function setupCustomSelect(wrapper) {
        const trigger = wrapper.querySelector('.custom-select-trigger');
        const optionsContainer = wrapper.querySelector('.custom-select-options');

        if (!trigger || !optionsContainer) {
            console.error('Custom select HTML structure is incorrect for:', wrapper);
            return;
        }

        trigger.addEventListener('click', (e) => {
            e.stopPropagation();
            document.querySelectorAll('.custom-select-wrapper.open').forEach(openWrapper => {
                if (openWrapper !== wrapper) {
                    openWrapper.classList.remove('open');
                }
            });
            wrapper.classList.toggle('open');
        });

        optionsContainer.addEventListener('click', (e) => {
            const option = e.target.closest('.custom-select-option');
            if (option && !option.classList.contains('custom-select-header')) {
                if (option.style.cursor === 'not-allowed') return;

                wrapper.dataset.value = option.dataset.value;
                trigger.textContent = option.textContent;

                optionsContainer.querySelectorAll('.custom-select-option').forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');

                wrapper.classList.remove('open');
            }
        });

        if (wrapper.id.includes('model-name')) {
            updateModelOptions(wrapper.id);
        }
    }

    document.addEventListener('click', () => {
        document.querySelectorAll('.custom-select-wrapper.open').forEach(openWrapper => {
            openWrapper.classList.remove('open');
        });
    });

    // --- DYNAMIC MODEL DROPDOWN WITH HEADERS ---
    function updateModelOptions(modelSelectId) {
        const modelWrapper = document.getElementById(modelSelectId);
        if (!modelWrapper) return;

        const optionsContainer = modelWrapper.querySelector('.custom-select-options');
        const trigger = modelWrapper.querySelector('.custom-select-trigger');

        optionsContainer.innerHTML = '';

        Object.keys(modelOptions).forEach(category => {
            const headerDiv = document.createElement('div');
            headerDiv.className = 'custom-select-header';
            headerDiv.textContent = category;
            optionsContainer.appendChild(headerDiv);

            const models = modelOptions[category];
            models.forEach(model => {
                const optionDiv = document.createElement('div');
                optionDiv.className = 'custom-select-option';
                optionDiv.dataset.value = model;
                optionDiv.textContent = model;
                optionsContainer.appendChild(optionDiv);
            });
        });

        const firstOption = optionsContainer.querySelector('.custom-select-option:not(.custom-select-header)');
        if (firstOption) {
            firstOption.classList.add('selected');
            trigger.textContent = firstOption.textContent;
            modelWrapper.dataset.value = firstOption.dataset.value;
        } else {
             trigger.textContent = 'Select a Model';
             modelWrapper.dataset.value = '';
        }
    }

    // --- UI/HELPER FUNCTIONS ---
    function toggleExpand(contentId, button) {
        document.querySelectorAll('.expand-content').forEach(content => {
            if (content.id !== contentId) {
                content.classList.add('hidden');
                content.previousElementSibling.classList.remove('active');
            }
        });
        const content = document.getElementById(contentId);
        content.classList.toggle('hidden');
        button.classList.toggle('active');
        if (!content.classList.contains('hidden')) {
            content.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }
    window.toggleExpand = toggleExpand;

    function toggleMenu() {
        menuDropdown.classList.toggle("hidden");
        menuToggle.setAttribute("aria-expanded", String(!menuDropdown.classList.contains("hidden")));
    }

    function closeMenuOnClickOutside(e) {
        if (!menuDropdown.contains(e.target) && !menuToggle.contains(e.target)) {
            menuDropdown.classList.add("hidden");
            menuToggle.setAttribute("aria-expanded", "false");
        }
    }

    function handleSmoothScroll(e) {
        e.preventDefault();
        const targetId = e.currentTarget.dataset.target;
        const target = document.getElementById(targetId);
        if (target) {
            const offset = (document.getElementById("site-navbar")?.offsetHeight || 0) + 8;
            const top = target.getBoundingClientRect().top + window.scrollY - offset;
            window.scrollTo({ top, behavior: "smooth" });
        }
        menuDropdown.classList.add("hidden");
        menuToggle.setAttribute("aria-expanded", "false");
    }

    function showToast(message, isError = false) {
        console.log(`TOAST: ${message}`);
        alert(message);
    }

    // --- FILE HANDLING ---
    function handleFileChange(e) {
        const input = e.target;
        const file = input.files[0];
        const labelId = input.closest('.file-drop-area').dataset.labelId;
        const label = document.getElementById(labelId);
        if (file) {
            label.textContent = file.name;
        }
    }

    function handleFileDrop(e) {
        e.preventDefault();
        const area = e.currentTarget;
        area.classList.remove('border-purple-500', 'bg-lavender');
        const inputId = area.dataset.inputId;
        const input = document.getElementById(inputId);
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        }
    }

    // --- CHATBOT LOGIC ---
    function toggleChatbotWindow() {
        chatbotWindow.classList.toggle('open');
        if (chatbotWindow.classList.contains('open')) {
            if (chatMessages.children.length === 0) {
                 const welcomeMessage = "Welcome to the AutoML Assistant! How can I help you today?";
                addChatMessage(welcomeMessage, false);
            }
            chatInput.focus();
        }
    }

    function closeChatbotOnClickOutside(e) {
        if (!chatbotWindow.contains(e.target) && !chatbotBubble.contains(e.target)) {
            chatbotWindow.classList.remove('open');
        }
    }

    function addChatMessage(message, isUser) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message-item ${isUser ? 'user-message' : 'bot-message'}`;
        messageDiv.innerHTML = `<p>${message}</p>`;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'chat-message-item bot-message typing-indicator';
        indicator.innerHTML = `<div class="typing-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
        chatMessages.appendChild(indicator);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return indicator;
    }

    async function sendChatMessage() {
        const message = chatInput.value.trim();
        if (!message) return;

        addChatMessage(message, true);
        chatInput.value = '';
        const typingIndicator = showTypingIndicator();

        try {
            const response = await fetch(`${API_BASE_URL}/gemini/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: message })
            });

            chatMessages.removeChild(typingIndicator);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Failed to get response from AI assistant.');
            }

            const data = await response.json();
            addChatMessage(data.reply, false);

        } catch (error) {
            console.error("Chatbot API error:", error);
            addChatMessage(`Sorry, I encountered an error: ${error.message}`, false);
        }
    }

    // --- TABULAR ANALYSIS LOGIC ---
    async function startTabularAnalysis(endpoint, payload, isFormData = false) {
        clearResults();
        showProcessingView();

        const fetchOptions = {
            method: 'POST',
            body: isFormData ? payload : JSON.stringify(payload)
        };
        if (!isFormData) {
            fetchOptions.headers = { 'Content-Type': 'application/json' };
        }

        try {
            const response = await fetch(`${API_BASE_URL}/tabular${endpoint}`, fetchOptions);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Analysis request failed.');
            }
            const data = await response.json();
            activeJobId = data.job_id;
            pollJobStatus(activeJobId, `${API_BASE_URL}/tabular/status/`, `${API_BASE_URL}/tabular/results/`);
        } catch (error) {
            console.error('Tabular analysis failed:', error);
            showToast(error.message, true);
            clearResults();
        }
    }

    function handleTabularFileUpload() {
        const fileInput = document.getElementById('file-upload-input');
        if (!fileInput.files.length) {
            return showToast('Please select a file to upload.', true);
        }
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('model_name', document.getElementById('upload-model-name').dataset.value);
        formData.append('problem_type', document.getElementById('upload-problem-type').dataset.value);
        formData.append('target_column', document.getElementById('upload-target-column').value);

        startTabularAnalysis('/analyze-file', formData, true);
    }

    function handleTabularUrlManual() {
        const payload = {
            url: document.getElementById('manual-url').value,
            model_name: document.getElementById('manual-model-name').dataset.value,
            problem_type: document.getElementById('manual-problem-type').dataset.value,
            target_column: document.getElementById('manual-target-column').value,
        };
        if (!payload.url || !payload.target_column) {
            return showToast('URL and Target Column are required.', true);
        }
        startTabularAnalysis('/analyze-url', payload);
    }

    function handleTabularUrlAutoML() {
        const payload = {
            url: document.getElementById('automl-url').value,
            target_column: document.getElementById('automl-target-column').value,
        };
        if (!payload.url || !payload.target_column) {
            return showToast('URL and Target Column are required.', true);
        }
        startTabularAnalysis('/automl', payload);
    }

    // --- IMAGE CLASSIFICATION LOGIC ---
    async function handleImageTrain() {
        const fileInput = document.getElementById('image-zip-input');
        const statusDiv = document.getElementById('image-training-status');

        if (!fileInput.files.length) {
            return showToast('Please select a ZIP file to upload.', true);
        }

        statusDiv.textContent = 'Uploading dataset...';
        statusDiv.classList.remove('hidden', 'text-green-600', 'text-red-600');
        statusDiv.classList.add('text-blue-600');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const uploadResponse = await fetch(`${API_BASE_URL}/image/upload-dataset`, { method: 'POST', body: formData });
            if (!uploadResponse.ok) throw new Error('Dataset upload failed.');
            const uploadData = await uploadResponse.json();

            statusDiv.textContent = 'Starting model training...';

            const trainResponse = await fetch(`${API_BASE_URL}/image/train-model`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ dataset_id: uploadData.dataset_id })
            });
            if (!trainResponse.ok) throw new Error('Failed to start training job.');
            const trainData = await trainResponse.json();
            activeModelId = trainData.model_id;

            pollJobStatus(activeModelId, `${API_BASE_URL}/image/model-status/`, null, true);

        } catch (error) {
            console.error('Image training failed:', error);
            statusDiv.textContent = `Error: ${error.message}`;
            statusDiv.classList.add('text-red-600');
        }
    }

    async function handleImagePredict() {
        const fileInput = document.getElementById('image-predict-input');
        const resultDiv = document.getElementById('image-prediction-result');

        if (!fileInput.files.length) {
            return showToast('Please select an image to predict.', true);
        }
        if (!activeModelId) {
            return showToast('No model has been trained yet.', true);
        }

        resultDiv.textContent = 'Predicting...';
        resultDiv.classList.remove('hidden', 'text-green-600', 'text-red-600');
        resultDiv.classList.add('text-blue-600');

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch(`${API_BASE_URL}/image/predict-image/${activeModelId}`, { method: 'POST', body: formData });
            if (!response.ok) {
                 const error = await response.json();
                 throw new Error(error.detail || 'Prediction failed.');
            }
            const data = await response.json();

            resultDiv.textContent = `Predicted Class: ${data.predicted_class} (Confidence: ${(data.confidence * 100).toFixed(2)}%)`;
            resultDiv.classList.add('text-green-600');

        } catch (error) {
            console.error('Image prediction failed:', error);
            resultDiv.textContent = `Error: ${error.message}`;
            resultDiv.classList.add('text-red-600');
        }
    }

    // --- POLLING & RESULTS LOGIC ---
    function pollJobStatus(id, statusUrl, resultsUrl, isImageJob = false) {
        clearInterval(jobPollingInterval);

        jobPollingInterval = setInterval(async () => {
            try {
                const response = await fetch(`${statusUrl}${id}`);
                if (!response.ok) {
                    throw new Error('Could not fetch job status.');
                }
                const data = await response.json();

                updateProcessingView(data.status);

                if (isImageJob) {
                    const statusDiv = document.getElementById('image-training-status');
                    statusDiv.textContent = `Status: ${data.status}`;
                    if (data.status === 'ready') {
                        clearInterval(jobPollingInterval);
                        statusDiv.textContent = 'Training complete! Model is ready for prediction.';
                        statusDiv.classList.remove('text-blue-600');
                        statusDiv.classList.add('text-green-600');
                        document.getElementById('image-prediction-section').classList.remove('opacity-50', 'pointer-events-none');
                    } else if (data.status === 'failed') {
                         clearInterval(jobPollingInterval);
                         statusDiv.textContent = 'Training failed. Check server logs.';
                         statusDiv.classList.add('text-red-600');
                    }
                } else {
                    if (data.status === 'complete') {
                        clearInterval(jobPollingInterval);
                        const resultsResponse = await fetch(`${resultsUrl}${id}`);
                        const results = await resultsResponse.json();
                        displayTabularResults(results);
                    } else if (data.status === 'failed') {
                        clearInterval(jobPollingInterval);
                        showToast(`Job ${id} failed: ${data.error}`, true);
                        clearResults();
                    }
                }
            } catch (error) {
                console.error('Polling error:', error);
                clearInterval(jobPollingInterval);
                showToast('Error checking job status.', true);
                clearResults();
            }
        }, 2000);
    }

    function clearResults() {
        dynamicContentArea.innerHTML = '';
        clearInterval(jobPollingInterval);
        activeJobId = null;
    }

    function showProcessingView() {
        const html = `
            <div id="progressSectionContent" class="card-modern">
                <h2 class="text-3xl font-bold text-gray-900 mb-6 text-center">Analyzing Your Data...</h2>
                <p class="text-lg text-gray-600 mb-8 text-center">Our AI is hard at work. Please wait.</p>
                <div class="space-y-6 max-w-2xl mx-auto">
                    <div id="processing-status-text" class="text-center font-semibold text-xl text-purple-600">Status: Pending</div>
                    <div class="w-full bg-gray-200 rounded-full h-4"><div id="progress-bar" class="bg-medium-slateblue h-4 rounded-full transition-all duration-500" style="width: 5%"></div></div>
                </div>
            </div>`;
        dynamicContentArea.innerHTML = html;
        dynamicContentArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function updateProcessingView(status) {
        const statusText = document.getElementById('processing-status-text');
        const progressBar = document.getElementById('progress-bar');
        if(!statusText || !progressBar) return;

        statusText.textContent = `Status: ${status}`;
        const progressMap = {
            'pending': 5, 'downloading': 15, 'loading_data': 25,
            'preprocessing': 40, 'training_Random_Forest': 60,
            'training_Logistic_Regression': 60, 'training_Decision_Tree': 60,
            'evaluating': 85, 'complete': 100
        };
        progressBar.style.width = `${progressMap[status] || 10}%`;
    }

    function displayTabularResults(results) {
        const isAutoML = 'best_model' in results;
        const mainResults = isAutoML ? results.best_model : results;
        const comparisonData = isAutoML ? results.comparison : [mainResults];
        const problemType = mainResults.problem_type;

        let tableHeaders = '';
        let tableRows = '';

        if (problemType === 'Classification') {
            tableHeaders = `
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy</th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">F1 Score</th>
            `;
            comparisonData.sort((a, b) => b.metrics.accuracy - a.metrics.accuracy).forEach(model => {
                const isBest = model.model_name === mainResults.model_name;
                tableRows += `<tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium ${isBest && isAutoML ? 'text-medium-slateblue font-bold' : 'text-gray-900'}">
                        ${model.model_name} ${isBest && isAutoML ? '<span class="ml-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">Best</span>' : ''}
                    </td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm">${(model.metrics.accuracy * 100).toFixed(2)}%</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm">${model.metrics.f1_score.toFixed(3)}</td>
                </tr>`;
            });
        } else { // Regression
             tableHeaders = `
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R-squared</th>
                <th scope="col" class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Mean Squared Error</th>
            `;
            comparisonData.sort((a, b) => b.metrics.r2_score - a.metrics.r2_score).forEach(model => {
                 tableRows += `<tr>
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${model.model_name}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm">${model.metrics.r2_score.toFixed(3)}</td>
                    <td class="px-6 py-4 whitespace-nowrap text-sm">${model.metrics.mean_squared_error.toFixed(2)}</td>
                </tr>`;
            });
        }

        const interpretationHtml = isAutoML && problemType === 'Classification' ? `
            <div class="border-t border-gray-200 mt-6 pt-6">
                <button id="interpret-btn" class="btn-primary w-full">Interpret Results with AI</button>
                <div id="interpretation-text" class="mt-4 text-gray-700 hidden p-4 bg-gray-50 rounded-lg"></div>
            </div>` : '';

        const html = `
            <div id="resultsSectionContent">
                <h2 class="text-3xl font-bold text-gray-900 mb-6 text-center">Analysis Results</h2>
                <div class="card-modern mb-8">
                    <h3 class="text-xl font-semibold text-gray-900 mb-4">Summary</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-gray-700">
                        <div><p class="font-medium">Problem Type:</p><p class="text-medium-slateblue font-semibold">${mainResults.problem_type}</p></div>
                        <div><p class="font-medium">Target Column:</p><p class="text-medium-slateblue font-semibold">${mainResults.target_column}</p></div>
                    </div>
                </div>
                <div class="card-modern mb-8">
                    <h3 class="text-xl font-semibold text-gray-900 mb-4">${isAutoML ? 'Model Comparison & Interpretation' : 'Model Performance'}</h3>
                    <div class="overflow-x-auto mb-6">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead class="bg-gray-50"><tr>${tableHeaders}</tr></thead>
                            <tbody class="bg-white divide-y divide-gray-200">${tableRows}</tbody>
                        </table>
                    </div>
                    <div id="metricsChart" class="h-80"></div>
                    ${interpretationHtml}
                </div>
            </div>`;
        dynamicContentArea.innerHTML = html;

        initializeCharts(comparisonData, problemType);

        if (isAutoML && problemType === 'Classification') {
            document.getElementById('interpret-btn').addEventListener('click', interpretResults);
        }
    }

    async function interpretResults() {
        if (!activeJobId) return;
        const btn = document.getElementById('interpret-btn');
        const resultDiv = document.getElementById('interpretation-text');
        btn.disabled = true;
        btn.textContent = 'Interpreting...';
        resultDiv.classList.remove('hidden');
        resultDiv.textContent = 'AI is analyzing the results...';

        try {
            const response = await fetch(`${API_BASE_URL}/gemini/interpret-results/${activeJobId}`);
            if (!response.ok) {
                 const error = await response.json();
                 throw new Error(error.detail || 'Failed to get interpretation.');
            }
            const data = await response.json();
            resultDiv.innerHTML = data.interpretation;
        } catch (error) {
            resultDiv.textContent = `Error: ${error.message}`;
        } finally {
            btn.style.display = 'none';
        }
    }

    function initializeCharts(modelData, problemType) {
        let traces;
        const modelNames = modelData.map(m => m.model_name);

        if (problemType === 'Classification') {
            const accuracies = modelData.map(m => m.metrics.accuracy * 100);
            const f1Scores = modelData.map(m => m.metrics.f1_score);
            traces = [
                { x: modelNames, y: accuracies, name: 'Accuracy (%)', type: 'bar', marker: { color: 'var(--color-medium-slateblue)' }},
                { x: modelNames, y: f1Scores, name: 'F1 Score', type: 'bar', marker: { color: 'var(--color-medium-purple)' }}
            ];
        } else { // Regression
             const r2Scores = modelData.map(m => m.metrics.r2_score);
             traces = [{ x: modelNames, y: r2Scores, name: 'R-squared', type: 'bar', marker: { color: 'var(--color-medium-slateblue)' }}];
        }

        const layout = { title: 'Model Performance Metrics', barmode: 'group', yaxis: { title: 'Score' }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)' };

        Plotly.newPlot('metricsChart', traces, layout, { responsive: true });
    }
});