// Medical X-ray Classifier - Frontend JavaScript with Heatmap

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const filePreview = document.getElementById('filePreview');
    const fileInfo = document.getElementById('fileInfo');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const resultsContent = document.getElementById('resultsContent');
    const resultsPlaceholder = document.getElementById('resultsPlaceholder');
    const alertBox = document.getElementById('alertBox');
    const binaryPrediction = document.getElementById('binaryPrediction');
    const binaryConfidence = document.getElementById('binaryConfidence');
    const binaryConfidenceBar = document.getElementById('binaryConfidenceBar');
    const binaryConfidenceFill = document.getElementById('binaryConfidenceFill');
    const subtypeCard = document.getElementById('subtypeCard');
    const subtypePrediction = document.getElementById('subtypePrediction');
    const subtypeConfidence = document.getElementById('subtypeConfidence');
    const subtypeConfidenceBar = document.getElementById('subtypeConfidenceBar');
    const subtypeConfidenceFill = document.getElementById('subtypeConfidenceFill');
    const uploadedImage = document.getElementById('uploadedImage');
    const resultTimestamp = document.getElementById('resultTimestamp');
    
    // Heatmap elements
    const heatmapContainer = document.getElementById('heatmapContainer');
    const heatmapImage = document.getElementById('heatmapImage');
    const medicalAnalysis = document.getElementById('medicalAnalysis');
    
    // Current state
    let currentFile = null;
    
    // Event Listeners
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    fileInput.addEventListener('change', handleFileSelect);
    analyzeBtn.addEventListener('click', handleAnalyze);
    clearBtn.addEventListener('click', handleClear);
    
    // File handling functions
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('highlight');
    }
    
    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('highlight');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    function handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }
    
    function handleFile(file) {
        // Check file type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/gif'];
        if (!validTypes.includes(file.type)) {
            showAlert('Please upload a valid image file (JPEG, PNG, BMP, GIF).', 'error');
            return;
        }
        
        // Check file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showAlert('File size must be less than 10MB.', 'error');
            return;
        }
        
        currentFile = file;
        fileName.textContent = file.name;
        fileInfo.classList.add('active');
        
        // Create preview
        const reader = new FileReader();
        reader.onload = function(e) {
            filePreview.src = e.target.result;
            filePreview.classList.add('active');
            uploadedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);
        
        // Enable analyze button
        analyzeBtn.disabled = false;
        
        // Hide previous results
        hideResults();
        hideAlert();
    }
    
    // Analyze button handler
    async function handleAnalyze() {
        if (!currentFile) return;
        
        // Show loading state
        showLoading();
        
        // Prepare form data
        const formData = new FormData();
        formData.append('file', currentFile);
        
        try {
            // Send request to backend
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || 'Failed to analyze image');
            }
            
            // Display results
            displayResults(data);
            showAlert('Analysis complete!', 'success');
            
        } catch (error) {
            console.error('Error:', error);
            showAlert(error.message || 'An error occurred during analysis.', 'error');
        } finally {
            hideLoading();
        }
    }
    
    // Clear button handler
    function handleClear() {
        currentFile = null;
        fileInput.value = '';
        fileName.textContent = '';
        filePreview.src = '';
        filePreview.classList.remove('active');
        fileInfo.classList.remove('active');
        analyzeBtn.disabled = true;
        hideResults();
        hideAlert();
    }
    
    // Display results from backend
    function displayResults(data) {
        // Update binary prediction
        const binaryPred = data.binary.prediction;
        const binaryConf = data.binary.confidence * 100;
        
        binaryPrediction.textContent = binaryPred;
        binaryConfidence.textContent = `${binaryConf.toFixed(1)}%`;
        binaryConfidenceFill.style.width = `${binaryConf}%`;
        
        // Update confidence value above progress bar
        const binaryConfidenceValue = document.getElementById('binaryConfidenceValue');
        if (binaryConfidenceValue) {
            binaryConfidenceValue.textContent = `${binaryConf.toFixed(1)}%`;
        }
        
        // Set card class based on prediction
        const binaryCard = document.querySelector('.result-card:nth-child(1)');
        binaryCard.classList.remove('normal', 'pneumonia');
        binaryCard.classList.add(binaryPred.toLowerCase());
        
        // Set confidence color
        setConfidenceColor(binaryConfidence, binaryConf);
        setConfidenceBarColor(binaryConfidenceFill, binaryConf);
        
        // Update subtype if present
        if (data.subtype && data.subtype.prediction) {
            const subtypePred = data.subtype.prediction;
            const subtypeConf = data.subtype.confidence * 100;
            
            subtypePrediction.textContent = subtypePred;
            subtypeConfidence.textContent = `${subtypeConf.toFixed(1)}%`;
            subtypeConfidenceFill.style.width = `${subtypeConf}%`;
            
            // Update subtype confidence value above progress bar
            const subtypeConfidenceValue = document.getElementById('subtypeConfidenceValue');
            if (subtypeConfidenceValue) {
                subtypeConfidenceValue.textContent = `${subtypeConf.toFixed(1)}%`;
            }
            
            // Set confidence color for subtype
            setConfidenceColor(subtypeConfidence, subtypeConf);
            setConfidenceBarColor(subtypeConfidenceFill, subtypeConf);
            
            subtypeCard.style.display = 'block';
        } else {
            subtypeCard.style.display = 'none';
        }
        
        // Update heatmap if available
        if (data.heatmap_available && data.heatmap_url) {
            if (heatmapImage) {
                heatmapImage.src = data.heatmap_url + '?t=' + new Date().getTime();
                heatmapImage.style.display = 'block';
            }
            if (heatmapContainer) {
                heatmapContainer.style.display = 'block';
            }
        } else {
            if (heatmapImage) heatmapImage.style.display = 'none';
            if (heatmapContainer) heatmapContainer.style.display = 'none';
        }
        
        // Update medical analysis if available
        if (data.medical_analysis && medicalAnalysis) {
            medicalAnalysis.textContent = data.medical_analysis;
            medicalAnalysis.style.display = 'block';
        } else if (medicalAnalysis) {
            medicalAnalysis.style.display = 'none';
        }
        
        // Update uploaded image URL
        if (data.uploaded_image_url) {
            uploadedImage.src = data.uploaded_image_url + '?t=' + new Date().getTime();
        }
        
        // Update timestamp
        if (data.timestamp) {
            const date = new Date(data.timestamp);
            resultTimestamp.textContent = `Analyzed at ${date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})} on ${date.toLocaleDateString()}`;
        }
        
        // Show results section
        resultsPlaceholder.style.display = 'none';
        resultsContent.classList.add('active');
    }
    
    // Helper functions
    function setConfidenceColor(element, confidence) {
        element.classList.remove('confidence-high', 'confidence-medium', 'confidence-low');
        
        if (confidence >= 80) {
            element.classList.add('confidence-high');
        } else if (confidence >= 60) {
            element.classList.add('confidence-medium');
        } else {
            element.classList.add('confidence-low');
        }
    }
    
    function setConfidenceBarColor(element, confidence) {
        if (confidence >= 80) {
            element.style.backgroundColor = '#34a853'; // green
        } else if (confidence >= 60) {
            element.style.backgroundColor = '#f9ab00'; // orange
        } else {
            element.style.backgroundColor = '#ea4335'; // red
        }
    }
    
    function showLoading() {
        loadingOverlay.classList.add('active');
        analyzeBtn.disabled = true;
    }
    
    function hideLoading() {
        loadingOverlay.classList.remove('active');
        analyzeBtn.disabled = false;
    }
    
    function showAlert(message, type = 'error') {
        alertBox.textContent = message;
        alertBox.className = `alert alert-${type} active`;
    }
    
    function hideAlert() {
        alertBox.classList.remove('active');
    }
    
    function hideResults() {
        resultsContent.classList.remove('active');
        resultsPlaceholder.style.display = 'block';
        
        // Hide heatmap and analysis
        if (heatmapImage) heatmapImage.style.display = 'none';
        if (heatmapContainer) heatmapContainer.style.display = 'none';
        if (medicalAnalysis) medicalAnalysis.style.display = 'none';
    }
    
    // Initialize
    analyzeBtn.disabled = true;
});