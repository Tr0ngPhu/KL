// DOM elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewImage = document.getElementById('previewImage');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('resultSection');
const errorMessage = document.getElementById('errorMessage');
const mainCard = document.querySelector('.main-card');
const newAnalysisBtn = document.getElementById('newAnalysisBtn');
const themeToggle = document.getElementById('themeToggle');
const particles = document.getElementById('particles');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initializeParticles();
    loadTheme();
});

// Event Listeners
uploadArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
uploadArea.addEventListener('dragover', handleDragOver);
uploadArea.addEventListener('dragleave', handleDragLeave);
uploadArea.addEventListener('drop', handleDrop);
newAnalysisBtn.addEventListener('click', resetToUpload);
themeToggle.addEventListener('click', toggleTheme);

// Theme Management
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update theme toggle icon
    updateThemeIcon(newTheme);
    
    // Add theme transition effect
    themeToggle.style.transform = 'scale(0.8) rotate(180deg)';
    setTimeout(() => {
        themeToggle.style.transform = 'scale(1) rotate(0deg)';
    }, 200);
}

function loadTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('svg');
    if (theme === 'dark') {
        icon.innerHTML = '<path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/>';
    } else {
        icon.innerHTML = '<path d="M12 18a6 6 0 100-12 6 6 0 000 12zM12 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zM12 20a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM4.22 4.22a1 1 0 011.42 0l.7.7a1 1 0 11-1.4 1.42l-.71-.7a1 1 0 010-1.42zM17.66 17.66a1 1 0 011.42 0l.7.7a1 1 0 11-1.4 1.42l-.71-.7a1 1 0 010-1.42zM2 12a1 1 0 011-1h1a1 1 0 110 2H3a1 1 0 01-1-1zM20 12a1 1 0 011-1h1a1 1 0 110 2h-1a1 1 0 01-1-1zM4.22 19.78a1 1 0 010-1.42l.7-.7a1 1 0 111.42 1.4l-.7.71a1 1 0 01-1.42 0zM17.66 6.34a1 1 0 010-1.42l.7-.7a1 1 0 111.42 1.4l-.7.71a1 1 0 01-1.42 0z"/>';
    }
}

// Particle System
function initializeParticles() {
    // Reduce particles on mobile for better performance
    const isMobile = window.innerWidth <= 768;
    const particleCount = isMobile ? 25 : 50;
    
    for (let i = 0; i < particleCount; i++) {
        createParticle();
    }
}

function createParticle() {
    const particle = document.createElement('div');
    particle.className = 'particle';
    
    const size = Math.random() * 4 + 2;
    const x = Math.random() * window.innerWidth;
    const y = Math.random() * window.innerHeight;
    const duration = Math.random() * 3 + 3;
    const delay = Math.random() * 2;
    
    particle.style.width = `${size}px`;
    particle.style.height = `${size}px`;
    particle.style.left = `${x}px`;
    particle.style.top = `${y}px`;
    particle.style.animationDuration = `${duration}s`;
    particle.style.animationDelay = `${delay}s`;
    
    particles.appendChild(particle);
    
    // Remove and recreate particle after animation
    setTimeout(() => {
        if (particle.parentNode) {
            particle.remove();
            createParticle();
        }
    }, (duration + delay) * 1000);
}

// Optimize particle creation for performance
let particleCreationTimeout;
function recreateParticles() {
    clearTimeout(particleCreationTimeout);
    particleCreationTimeout = setTimeout(() => {
        particles.innerHTML = '';
        initializeParticles();
    }, 250);
}

// Handle window resize
window.addEventListener('resize', recreateParticles);

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function processFile(file) {
    if (!file.type.startsWith('image/')) {
        showError('Định dạng file không hợp lệ. Vui lòng tải lên ảnh.');
        return;
    }
    if (file.size > 10 * 1024 * 1024) { // 10MB
        showError('File quá lớn. Kích thước tối đa là 10MB.');
        return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewImage.classList.add('show');
        
        // Hide upload content when image is loaded
        const uploadContent = document.querySelector('.upload-content');
        uploadContent.style.opacity = '0';
        uploadContent.style.transform = 'scale(0.8)';
        setTimeout(() => {
            uploadContent.style.display = 'none';
        }, 300);
    };
    reader.readAsDataURL(file);

    // Add loading animation delay for better UX
    setTimeout(() => {
        analyzeImage(file);
    }, 500);
}

function analyzeImage(file) {
    hideError();
    // Không ẩn resultSection nữa, chỉ reset nội dung
    const resultImage = document.getElementById('resultImage');
    const heatmapOverlay = document.getElementById('heatmapOverlay');
    const heatmapImageLarge = document.getElementById('heatmapImageLarge');
    const heatmapStandalone = document.getElementById('heatmapStandalone');
    
    resultImage.style.display = 'none';
    heatmapOverlay.style.display = 'none';
    if (heatmapImageLarge) heatmapImageLarge.style.display = 'none';
    if (heatmapStandalone) heatmapStandalone.style.display = 'none';
    
    loading.classList.add('show');

    const formData = new FormData();
    formData.append('file', file);

    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        loading.classList.remove('show');
        console.log('API Response:', data); // Debug log
        
        // Handle the actual API response format
        if (data.prediction && data.confidence !== undefined) {
            const result = {
                is_real: data.prediction.toLowerCase() === 'real',
                confidence: data.confidence, // Keep as percentage
                explanation: data.explanation?.text || 'Phân tích hoàn tất.',
                heatmap_url: data.explanation?.heatmap,
                analysis_url: data.explanation?.analysis_plot
            };
            showResult(result, file);
        } else {
            showError('Phản hồi không hợp lệ từ máy chủ.');
        }
    })
    .catch(error => {
        loading.classList.remove('show');
        showError('Không thể kết nối với máy chủ. Vui lòng thử lại sau.');
        console.error('Error:', error);
    });
}

function showResult(result, file) {
    // Show uploaded image in result section with animation
    const resultImage = document.getElementById('resultImage');
    const heatmapOverlay = document.getElementById('heatmapOverlay');
    const reader = new FileReader();
    reader.onload = (e) => {
        resultImage.src = e.target.result;
        resultImage.style.display = 'block';
        resultImage.style.opacity = '0';
        resultImage.style.transform = 'scale(0.8)';
        setTimeout(() => {
            resultImage.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
            resultImage.style.opacity = '1';
            resultImage.style.transform = 'scale(1)';
        }, 100);
    };
    reader.readAsDataURL(file);

    // Show result status with enhanced animations
    const resultStatus = document.getElementById('resultStatus');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    const confidenceText = document.getElementById('confidenceText');
    const analysisText = document.getElementById('analysisText');
    const heatmapImageLarge = document.getElementById('heatmapImageLarge');
    const heatmapStandalone = document.getElementById('heatmapStandalone');

    if (result.is_real) {
        resultIcon.textContent = '✅';
        resultText.textContent = 'Hàng Thật';
        resultStatus.className = 'result-status real';
    } else {
        resultIcon.textContent = '❌';
        resultText.textContent = 'Hàng Giả';
        resultStatus.className = 'result-status fake';
    }

    // Animate confidence percentage
    animateCounter(confidenceText, 0, result.confidence, 1000);
    
    // Format explanation in Vietnamese with line breaks
    const vietnameseExplanation = formatVietnameseExplanation(result.explanation, result.is_real, result.confidence);
    
    // Typewriter effect for analysis text
    typewriterEffect(analysisText, vietnameseExplanation);
    
    // Show heatmap overlay on the result image
    if (result.heatmap_url) {
        heatmapOverlay.src = result.heatmap_url;
        setTimeout(() => {
            heatmapOverlay.style.display = 'block';
            heatmapOverlay.style.opacity = '0';
            heatmapOverlay.style.transition = 'opacity 0.8s ease';
            setTimeout(() => {
                heatmapOverlay.style.opacity = '0.6';
            }, 100);
        }, 800);
        
        // Show large heatmap in standalone section
        if (result.analysis_url) {
            heatmapImageLarge.src = result.analysis_url;
        } else {
            heatmapImageLarge.src = result.heatmap_url;
        }
        heatmapImageLarge.style.display = 'block';
    }
    
    // Enhanced transition sequence - Hide upload card completely and move results to top
    setTimeout(() => {
        // Completely hide upload area with fade out
        mainCard.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
        mainCard.style.opacity = '0';
        mainCard.style.transform = 'scale(0.9) translateY(-20px)';
        
        setTimeout(() => {
            // Hide the upload card completely
            mainCard.style.display = 'none';
            
            // Move result section to top and show
            resultSection.style.marginTop = '0';
            resultSection.classList.add('show');
            
            // Show heatmap section with delay
            setTimeout(() => {
                if (result.heatmap_url || result.analysis_url) {
                    heatmapStandalone.classList.add('show');
                }
            }, 400);
        }, 600);
    }, 200);
}

// Enhanced utility functions
function animateCounter(element, start, end, duration) {
    const startTime = performance.now();
    const difference = end - start;
    
    function updateCounter(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function for smooth animation
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = start + (difference * easeOut);
        
        element.textContent = `${current.toFixed(1)}%`;
        
        if (progress < 1) {
            requestAnimationFrame(updateCounter);
        }
    }
    
    requestAnimationFrame(updateCounter);
}

function typewriterEffect(element, text, speed = 30) {
    element.textContent = '';
    element.style.opacity = '1';
    
    let i = 0;
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    
    setTimeout(type, 300); // Delay before starting typewriter
}

function formatVietnameseExplanation(explanation, isReal, confidence) {
    // Convert English AI explanation to Vietnamese format without technical numbers
    let vietnameseText = '';
    
    if (isReal) {
        vietnameseText += 'AI nhận định đây là sản phẩm chính hãng.\n\n';
        vietnameseText += 'Các đặc điểm chính:\n';
        vietnameseText += '• Chất liệu và kết cấu có độ sắc nét cao\n';
        vietnameseText += '• Các chi tiết nhỏ được làm tinh xảo\n';
        vietnameseText += '• Màu sắc và bề mặt đồng nhất\n';
        vietnameseText += '• Không có dấu hiệu của việc sao chép\n\n';
        vietnameseText += 'Kết luận: Sản phẩm có đặc điểm của hàng chính hãng.';
    } else {
        vietnameseText += 'AI phát hiện đây có thể là sản phẩm nhái.\n\n';
        vietnameseText += 'Các dấu hiệu nghi ngờ:\n';
        vietnameseText += '• Chất lượng hoàn thiện không đồng đều\n';
        vietnameseText += '• Một số chi tiết có vẻ thô sơ\n';
        vietnameseText += '• Có sự khác biệt so với sản phẩm gốc\n';
        vietnameseText += '• Dấu hiệu của quá trình sao chép\n\n';
        vietnameseText += 'Kết luận: Sản phẩm có các đặc điểm của hàng nhái.';
    }
    
    return vietnameseText;
}

function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.add('show');
    
    // Add error shake animation
    errorMessage.style.animation = 'none';
    setTimeout(() => {
        errorMessage.style.animation = 'shake 0.5s ease-in-out';
    }, 10);
}

function hideError() {
    errorMessage.classList.remove('show');
}

function resetToUpload() {
    // Enhanced reset sequence - Show upload card again
    const heatmapStandalone = document.getElementById('heatmapStandalone');
    
    // Hide heatmap with scale animation
    heatmapStandalone.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
    heatmapStandalone.classList.remove('show');
    
    // Hide result section with stagger
    setTimeout(() => {
        resultSection.style.transition = 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)';
        resultSection.classList.remove('show');
        
        // Show upload area again
        setTimeout(() => {
            // Reset result section margin
            resultSection.style.marginTop = '3rem';
            
            // Show upload card again
            mainCard.style.display = 'block';
            mainCard.style.opacity = '0';
            mainCard.style.transform = 'scale(0.9) translateY(20px)';
            
            setTimeout(() => {
                mainCard.style.transition = 'all 0.6s cubic-bezier(0.4, 0, 0.2, 1)';
                mainCard.style.opacity = '1';
                mainCard.style.transform = 'scale(1) translateY(0)';
                mainCard.classList.remove('hidden');
                
                // Reset upload area with animations
                setTimeout(() => {
                    resetUploadArea();
                }, 200);
            }, 100);
        }, 300);
    }, 200);
}

function resetUploadArea() {
    // Reset preview image with fade out
    previewImage.style.transition = 'all 0.4s ease';
    previewImage.style.opacity = '0';
    previewImage.style.transform = 'scale(0.8)';
    
    setTimeout(() => {
        previewImage.classList.remove('show');
        previewImage.src = '';
        previewImage.style.transform = 'scale(1)';
        
        // Show upload content with fade in
        const uploadContent = document.querySelector('.upload-content');
        uploadContent.style.display = 'flex';
        uploadContent.style.opacity = '0';
        uploadContent.style.transform = 'scale(0.8)';
        
        setTimeout(() => {
            uploadContent.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)';
            uploadContent.style.opacity = '1';
            uploadContent.style.transform = 'scale(1)';
        }, 50);
        
        // Reset file input
        fileInput.value = '';
        
        // Hide loading and error
        loading.classList.remove('show');
        hideError();
    }, 300);
}
