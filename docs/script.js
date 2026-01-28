// ===================================
// SMOOTH SCROLL & NAVIGATION
// ===================================

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ===================================
// NAVBAR SCROLL EFFECT
// ===================================

let lastScroll = 0;
const navbar = document.querySelector('.nav');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;
    
    // Add shadow on scroll
    if (currentScroll > 50) {
        navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.3)';
    } else {
        navbar.style.boxShadow = 'none';
    }
    
    lastScroll = currentScroll;
});

// ===================================
// INTERSECTION OBSERVER FOR ANIMATIONS
// ===================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe all sections and cards
const animateElements = document.querySelectorAll('.section-header, .project-card, .skill-category, .about-content, .contact-content');
animateElements.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'all 0.6s ease';
    observer.observe(el);
});

// ===================================
// PARTICLE EFFECT FOR HERO
// ===================================

function createParticles() {
    const heroBackground = document.querySelector('.hero-background');
    const particleCount = 30;
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 3 + 1;
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        const duration = Math.random() * 20 + 10;
        const delay = Math.random() * 5;
        
        particle.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            background: rgba(0, 217, 255, 0.3);
            border-radius: 50%;
            left: ${posX}%;
            top: ${posY}%;
            animation: float ${duration}s ease-in-out ${delay}s infinite;
            pointer-events: none;
        `;
        
        heroBackground.appendChild(particle);
    }
}

// Add CSS animation for particles
const style = document.createElement('style');
style.textContent = `
    @keyframes float {
        0%, 100% {
            transform: translate(0, 0);
            opacity: 0;
        }
        50% {
            transform: translate(${Math.random() * 100 - 50}px, ${Math.random() * 100 - 50}px);
            opacity: 0.5;
        }
    }
`;
document.head.appendChild(style);

createParticles();

// ===================================
// DYNAMIC TYPING EFFECT (Optional Enhancement)
// ===================================

const roles = [
    'Data Scientist',
    'ML Engineer', 
    'Python Developer',
    'Data Analyst'
];

let roleIndex = 0;
let charIndex = 0;
let isDeleting = false;
const typingSpeed = 100;
const deletingSpeed = 50;
const pauseDuration = 2000;

const heroLabel = document.querySelector('.hero-label');
const originalText = heroLabel.textContent;

function typeRole() {
    const currentRole = roles[roleIndex];
    
    if (isDeleting) {
        heroLabel.textContent = currentRole.substring(0, charIndex - 1);
        charIndex--;
    } else {
        heroLabel.textContent = currentRole.substring(0, charIndex + 1);
        charIndex++;
    }
    
    let timeout = isDeleting ? deletingSpeed : typingSpeed;
    
    if (!isDeleting && charIndex === currentRole.length) {
        timeout = pauseDuration;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        roleIndex = (roleIndex + 1) % roles.length;
    }
    
    setTimeout(typeRole, timeout);
}

// Start typing effect after page load
setTimeout(typeRole, 1000);

// ===================================
// PROJECT STATS COUNTER ANIMATION
// ===================================

function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            element.textContent = formatNumber(target);
            clearInterval(timer);
        } else {
            element.textContent = formatNumber(Math.floor(current));
        }
    }, 16);
}

function formatNumber(num) {
    if (num >= 100000) {
        return (num / 1000).toFixed(0) + 'K+';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

// Animate counters when they come into view
const statObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('counted')) {
            entry.target.classList.add('counted');
            const numbers = entry.target.querySelectorAll('.stat-number');
            numbers.forEach(num => {
                const text = num.textContent;
                const value = parseInt(text.replace(/[^0-9]/g, ''));
                if (value) {
                    num.textContent = '0';
                    animateCounter(num, value);
                }
            });
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.hero-stats').forEach(stat => {
    statObserver.observe(stat);
});

// ===================================
// HOVER EFFECTS FOR PROJECT CARDS
// ===================================

document.querySelectorAll('.project-card').forEach(card => {
    card.addEventListener('mouseenter', function(e) {
        const image = this.querySelector('.project-image');
        image.style.transform = 'scale(1.05)';
        image.style.transition = 'transform 0.4s ease';
    });
    
    card.addEventListener('mouseleave', function(e) {
        const image = this.querySelector('.project-image');
        image.style.transform = 'scale(1)';
    });
});

// ===================================
// CURSOR TRAIL EFFECT (Optional)
// ===================================

let mouseX = 0;
let mouseY = 0;
let cursorX = 0;
let cursorY = 0;

document.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
});

function createCursor() {
    const cursor = document.createElement('div');
    cursor.style.cssText = `
        position: fixed;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(0, 217, 255, 0.5);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        transition: transform 0.15s ease;
        display: none;
    `;
    document.body.appendChild(cursor);
    
    // Only show on desktop
    if (window.innerWidth > 768) {
        cursor.style.display = 'block';
    }
    
    function updateCursor() {
        cursorX += (mouseX - cursorX) * 0.1;
        cursorY += (mouseY - cursorY) * 0.1;
        cursor.style.transform = `translate(${cursorX - 10}px, ${cursorY - 10}px)`;
        requestAnimationFrame(updateCursor);
    }
    
    updateCursor();
}

createCursor();

// ===================================
// SKILL TAGS ANIMATION ON HOVER
// ===================================

document.querySelectorAll('.skill-item').forEach(skill => {
    skill.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-3px) scale(1.05)';
    });
    
    skill.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// ===================================
// PERFORMANCE OPTIMIZATION
// ===================================

// Lazy load images when implemented
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    imageObserver.unobserve(img);
                }
            }
        });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

// ===================================
// CONSOLE EASTER EGG
// ===================================

console.log('%c👋 Hey there!', 'font-size: 20px; font-weight: bold; color: #00D9FF;');
console.log('%cLooking at the code? I like your style!', 'font-size: 14px; color: #9BA9C1;');
console.log('%cFeel free to reach out: shouryat32@gmail.com', 'font-size: 14px; color: #FF6B35;');

// ===================================
// SCROLL PROGRESS INDICATOR
// ===================================

function createScrollProgress() {
    const progress = document.createElement('div');
    progress.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        height: 3px;
        background: linear-gradient(90deg, #00D9FF, #7B2CBF);
        width: 0%;
        z-index: 9999;
        transition: width 0.1s ease;
    `;
    document.body.appendChild(progress);
    
    window.addEventListener('scroll', () => {
        const windowHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (window.pageYOffset / windowHeight) * 100;
        progress.style.width = scrolled + '%';
    });
}

createScrollProgress();

console.log('%c✨ Portfolio loaded successfully!', 'font-size: 12px; color: #00D9FF;');
