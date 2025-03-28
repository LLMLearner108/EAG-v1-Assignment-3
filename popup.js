document.addEventListener('DOMContentLoaded', function() {
    const API_URL = 'http://localhost:8000';
    const taskTypeSelect = document.getElementById('taskType');
    const topKInput = document.getElementById('topKInput');
    const scoreInput = document.getElementById('scoreInput');
    const emailInput = document.getElementById('emailInput');
    const submitBtn = document.getElementById('submitBtn');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error');
    const resultDiv = document.getElementById('result');

    // Show/hide relevant inputs based on task type
    taskTypeSelect.addEventListener('change', function() {
        const taskType = this.value;
        topKInput.style.display = taskType === 'top_k' ? 'flex' : 'none';
        scoreInput.style.display = (taskType === 'score_filter' || taskType === 'email') ? 'flex' : 'none';
        emailInput.style.display = taskType === 'email' ? 'flex' : 'none';
    });

    // Initial display setup
    taskTypeSelect.dispatchEvent(new Event('change'));

    submitBtn.addEventListener('click', async function() {
        // Reset UI
        errorDiv.textContent = '';
        resultDiv.innerHTML = '';
        loadingDiv.style.display = 'block';
        submitBtn.disabled = true;

        try {
            const query = document.getElementById('query').value;
            const taskType = taskTypeSelect.value;
            const requestBody = {
                query: query,
                task_type: taskType
            };

            // Add task-specific parameters
            switch (taskType) {
                case 'top_k':
                    const topK = parseInt(document.getElementById('topK').value);
                    if (isNaN(topK) || topK < 1) {
                        throw new Error('Please enter a valid number for Top K');
                    }
                    requestBody.top_k = topK;
                    break;

                case 'score_filter':
                case 'email':
                    const minScore = parseFloat(document.getElementById('minScore').value);
                    if (isNaN(minScore) || minScore < 0 || minScore > 10) {
                        throw new Error('Please enter a valid score between 0 and 10');
                    }
                    requestBody.min_score = minScore;

                    if (taskType === 'email') {
                        const email = document.getElementById('recipientEmail').value;
                        if (!email || !email.includes('@')) {
                            throw new Error('Please enter a valid email address');
                        }
                        requestBody.recipient_email = email;
                    }
                    break;
            }

            // Make API request
            const response = await fetch(`${API_URL}/agent`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestBody)
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.detail || 'An error occurred');
            }

            // Display results
            if (taskType === 'email') {
                resultDiv.innerHTML = `
                    <div class="anime-item">
                        <strong>Email sent successfully!</strong><br>
                        Sent ${data.result.length} anime to ${requestBody.recipient_email}
                    </div>
                `;
            } else {
                resultDiv.innerHTML = data.result.map(anime => `
                    <div class="anime-item">
                        <strong>${anime.title}</strong><br>
                        Score: ${anime.score}
                    </div>
                `).join('');
            }

        } catch (error) {
            errorDiv.textContent = error.message;
        } finally {
            loadingDiv.style.display = 'none';
            submitBtn.disabled = false;
        }
    });
}); 