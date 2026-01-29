// ===================================
// SMOOTH SCROLL
// ===================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            const navHeight = document.querySelector('.nav').offsetHeight;
            const targetPosition = target.offsetTop - navHeight;
            window.scrollTo({
                top: targetPosition,
                behavior: 'smooth'
            });

            // Close mobile menu if open
            closeMobileMenu();
        }
    });
});

// ===================================
// SCROLL PROGRESS BAR
// ===================================

const navProgress = document.querySelector('.nav-progress');

function updateScrollProgress() {
    const winScroll = document.documentElement.scrollTop || document.body.scrollTop;
    const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    const scrolled = (winScroll / height) * 100;

    if (navProgress) {
        navProgress.style.width = scrolled + '%';
    }
}

window.addEventListener('scroll', updateScrollProgress);

// ===================================
// NAVBAR SCROLL EFFECT
// ===================================

const navbar = document.querySelector('.nav');
let lastScroll = 0;

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll > 50) {
        navbar.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.1)';
    } else {
        navbar.style.boxShadow = 'none';
    }

    lastScroll = currentScroll;
});

// ===================================
// MOBILE MENU
// ===================================

const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const mobileMenuOverlay = document.querySelector('.mobile-menu-overlay');

function openMobileMenu() {
    mobileMenuToggle.classList.add('active');
    mobileMenuOverlay.classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeMobileMenu() {
    mobileMenuToggle.classList.remove('active');
    mobileMenuOverlay.classList.remove('active');
    document.body.style.overflow = '';
}

if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', () => {
        if (mobileMenuOverlay.classList.contains('active')) {
            closeMobileMenu();
        } else {
            openMobileMenu();
        }
    });
}

// Close mobile menu when clicking on links
document.querySelectorAll('.mobile-menu-content a').forEach(link => {
    link.addEventListener('click', closeMobileMenu);
});

// Close mobile menu on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        closeMobileMenu();
        closeVizModal();
    }
});

// ===================================
// FADE IN ON SCROLL
// ===================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe elements (excluding project cards and viz cards which have their own staggered animation)
document.querySelectorAll('.featured-project, .skill-category, .timeline-item, .achievement-card').forEach(el => {
    observer.observe(el);
});

// ===================================
// ENHANCED STAGGERED CARD ANIMATION
// ===================================

const staggeredCardsObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (entry.isIntersecting) {
            // Calculate stagger delay based on card position in its grid
            const isProjectCard = entry.target.classList.contains('project-card');
            const isVizCard = entry.target.classList.contains('viz-card');

            let cards;
            if (isProjectCard) {
                cards = document.querySelectorAll('.project-card');
            } else if (isVizCard) {
                cards = document.querySelectorAll('.viz-card');
            }

            const cardIndex = Array.from(cards).indexOf(entry.target);

            // Apply staggered delay
            entry.target.style.transitionDelay = `${cardIndex * 100}ms`;
            entry.target.classList.add('card-revealed');

            // Clean up after animation
            setTimeout(() => {
                entry.target.style.transitionDelay = '0ms';
            }, 600 + (cardIndex * 100));

            staggeredCardsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.15, rootMargin: '0px 0px -30px 0px' });

// Set initial hidden state and observe project cards and viz cards
document.querySelectorAll('.project-card, .viz-card').forEach(card => {
    card.classList.add('card-hidden');
    staggeredCardsObserver.observe(card);
});

// ===================================
// COUNTER ANIMATION
// ===================================

function animateCounter(element) {
    const text = element.textContent;
    const hasPlus = text.includes('+');
    const hasPercent = text.includes('%');
    const hasGB = text.includes('GB');
    const value = parseInt(text.replace(/[^0-9]/g, ''));

    if (!value) return;

    const duration = 2000;
    const steps = 60;
    const increment = value / steps;
    let current = 0;
    let step = 0;

    const timer = setInterval(() => {
        current += increment;
        step++;

        if (step >= steps) {
            clearInterval(timer);
            current = value;
        }

        let displayValue = Math.floor(current);
        if (text.includes('K')) {
            displayValue = (current / 1000).toFixed(0) + 'K';
        }
        if (hasPlus) displayValue += '+';
        if (hasPercent) displayValue += '%';
        if (hasGB) displayValue = Math.floor(current) + 'GB';

        element.textContent = displayValue;
    }, duration / steps);
}

// Animate counters when they come into view
const counterObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting && !entry.target.classList.contains('animated')) {
            entry.target.classList.add('animated');
            animateCounter(entry.target);
        }
    });
}, { threshold: 0.5 });

document.querySelectorAll('.metric-value, .featured-result-value, .metric-highlight').forEach(counter => {
    counterObserver.observe(counter);
});

// ===================================
// VISUALIZATION FILTERS
// ===================================

const vizFilters = document.querySelectorAll('.viz-filter');
const vizCards = document.querySelectorAll('.viz-card');

vizFilters.forEach(filter => {
    filter.addEventListener('click', () => {
        // Update active state
        vizFilters.forEach(f => f.classList.remove('active'));
        filter.classList.add('active');

        // Filter cards
        const category = filter.dataset.filter;

        vizCards.forEach(card => {
            if (category === 'all' || card.dataset.category === category) {
                card.style.display = '';
                card.style.opacity = '0';
                setTimeout(() => {
                    card.style.opacity = '1';
                }, 50);
            } else {
                card.style.display = 'none';
            }
        });
    });
});

// ===================================
// VISUALIZATION MODAL
// ===================================

const vizModal = document.getElementById('vizModal');
const vizModalBody = document.getElementById('vizModalBody');
const vizModalClose = document.querySelector('.viz-modal-close');
const vizExpandButtons = document.querySelectorAll('.viz-card-expand');

function openVizModal(tableauUrl) {
    if (vizModal && vizModalBody && tableauUrl) {
        // Clear previous content
        vizModalBody.innerHTML = '';

        // Create tableau-viz element
        const tableauViz = document.createElement('tableau-viz');
        tableauViz.setAttribute('id', 'modal-viz');
        tableauViz.setAttribute('src', tableauUrl);
        tableauViz.setAttribute('width', '100%');
        tableauViz.setAttribute('height', '700');
        tableauViz.setAttribute('hide-tabs', '');
        tableauViz.setAttribute('toolbar', 'bottom');

        // Add loading indicator
        const loadingDiv = document.createElement('div');
        loadingDiv.className = 'viz-loading';
        loadingDiv.innerHTML = '<p>Loading visualization...</p>';
        vizModalBody.appendChild(loadingDiv);

        // Append tableau viz
        vizModalBody.appendChild(tableauViz);

        // Remove loading indicator after a delay
        setTimeout(() => {
            if (loadingDiv.parentNode) {
                loadingDiv.remove();
            }
        }, 2000);

        vizModal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }
}

function closeVizModal() {
    if (vizModal) {
        vizModal.classList.remove('active');
        document.body.style.overflow = '';

        // Clear the modal body to stop Tableau from loading
        if (vizModalBody) {
            vizModalBody.innerHTML = '';
        }
    }
}

// Handle viz card expand button clicks
vizExpandButtons.forEach(btn => {
    btn.addEventListener('click', (e) => {
        e.stopPropagation();

        // Find the parent viz-card and get the Tableau URL
        const vizCard = btn.closest('.viz-card');
        const tableauUrl = vizCard ? vizCard.dataset.vizUrl : null;
        const openFullscreen = vizCard ? vizCard.dataset.fullscreen === 'true' : false;

        if (tableauUrl) {
            if (openFullscreen) {
                // Open in new tab for fullscreen experience
                window.open(tableauUrl, '_blank');
            } else {
                openVizModal(tableauUrl);
            }
        } else {
            console.warn('No Tableau URL found for this visualization');
        }
    });
});

// Also allow clicking on the entire viz-card to open modal or fullscreen
document.querySelectorAll('.viz-card[data-viz-url]').forEach(card => {
    card.addEventListener('click', (e) => {
        // Don't trigger if clicking the expand button (it has its own handler)
        if (e.target.closest('.viz-card-expand')) return;

        const tableauUrl = card.dataset.vizUrl;
        const openFullscreen = card.dataset.fullscreen === 'true';

        if (tableauUrl) {
            if (openFullscreen) {
                // Open in new tab for fullscreen experience
                window.open(tableauUrl, '_blank');
            } else {
                openVizModal(tableauUrl);
            }
        }
    });

    // Add cursor pointer style
    card.style.cursor = 'pointer';
});

if (vizModalClose) {
    vizModalClose.addEventListener('click', closeVizModal);
}

// Close modal when clicking outside
if (vizModal) {
    vizModal.addEventListener('click', (e) => {
        if (e.target === vizModal) {
            closeVizModal();
        }
    });
}

// ===================================
// ACTIVE NAVIGATION
// ===================================

const sections = document.querySelectorAll('section[id]');
const navLinks = document.querySelectorAll('.nav-menu a');

window.addEventListener('scroll', () => {
    let current = '';
    const navHeight = document.querySelector('.nav').offsetHeight;

    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        if (window.pageYOffset >= (sectionTop - navHeight - 50)) {
            current = section.getAttribute('id');
        }
    });

    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// ===================================
// STAGGERED ANIMATIONS
// ===================================

function addStaggeredAnimations() {
    // Stagger featured highlights
    document.querySelectorAll('.featured-highlight').forEach((el, i) => {
        el.style.animationDelay = `${i * 0.1}s`;
    });

    // Stagger skill categories
    document.querySelectorAll('.skill-category').forEach((el, i) => {
        el.style.animationDelay = `${i * 0.1}s`;
    });

    // Stagger timeline items
    document.querySelectorAll('.timeline-item').forEach((el, i) => {
        el.style.animationDelay = `${i * 0.15}s`;
    });

    // Stagger viz cards
    document.querySelectorAll('.viz-card').forEach((el, i) => {
        el.style.animationDelay = `${i * 0.1}s`;
    });
}

// ===================================
// PERFORMANCE OPTIMIZATION
// ===================================

// Lazy load images
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
// HOVER EFFECTS
// ===================================

document.querySelectorAll('.featured-link, .project-card-link').forEach(link => {
    link.addEventListener('mouseenter', function() {
        this.style.transform = 'translateX(8px)';
    });

    link.addEventListener('mouseleave', function() {
        this.style.transform = 'translateX(0)';
    });
});

// ===================================
// ARCHITECTURE DIAGRAM ANIMATION
// ===================================

const archNodes = document.querySelectorAll('.arch-node');
const archConnectors = document.querySelectorAll('.arch-connector');

function animateArchitecture() {
    archNodes.forEach((node, i) => {
        node.style.opacity = '0';
        node.style.transform = 'translateY(20px)';

        setTimeout(() => {
            node.style.transition = 'all 0.5s ease';
            node.style.opacity = '1';
            node.style.transform = 'translateY(0)';
        }, i * 200);
    });

    archConnectors.forEach((connector, i) => {
        connector.style.opacity = '0';
        connector.style.transform = 'scaleY(0)';

        setTimeout(() => {
            connector.style.transition = 'all 0.3s ease';
            connector.style.opacity = '1';
            connector.style.transform = 'scaleY(1)';
        }, (i + 1) * 200 + 100);
    });
}

// Animate architecture diagram when visible
const archDiagram = document.querySelector('.architecture-diagram');
if (archDiagram) {
    const archObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateArchitecture();
                archObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });

    archObserver.observe(archDiagram);
}

// ===================================
// SMOOTH PAGE LOAD
// ===================================

window.addEventListener('load', () => {
    document.body.style.opacity = '0';

    setTimeout(() => {
        document.body.style.transition = 'opacity 0.3s ease';
        document.body.style.opacity = '1';
        addStaggeredAnimations();
    }, 10);
});

// ===================================
// TABLEAU HANDLING
// ===================================

document.addEventListener('DOMContentLoaded', () => {
    const tableauVizElements = document.querySelectorAll('tableau-viz');

    tableauVizElements.forEach(viz => {
        if (viz.getAttribute('src') === 'YOUR_TABLEAU_PUBLIC_URL_HERE') {
            // Visualization placeholder - will be shown via CSS
        }
    });
});

// Debug helper
window.checkTableauViz = function() {
    const vizElements = document.querySelectorAll('tableau-viz');
    console.log(`Found ${vizElements.length} Tableau visualizations`);

    vizElements.forEach((viz, index) => {
        const src = viz.getAttribute('src');
        console.log(`Viz ${index + 1}:`, src);

        if (src === 'YOUR_TABLEAU_PUBLIC_URL_HERE') {
            console.warn(`Viz ${index + 1} needs a real Tableau Public URL`);
        }
    });
};

// ===================================
// PHOENIX GALLERY MODAL
// ===================================

const phoenixImages = [
    { src: 'assets/pheonix/dashborad_landing_page.png', title: 'Dashboard Landing Page', desc: 'Main overview of the Phoenix Australia Survey Analytics Dashboard' },
    { src: 'assets/pheonix/show_classification.png', title: 'Question Classification', desc: 'AI-powered Kirkpatrick model classification of survey questions' },
    { src: 'assets/pheonix/ediy_classifications.png', title: 'Edit Classifications', desc: 'Human-in-the-loop interface for correcting classifications' },
    { src: 'assets/pheonix/participation.png', title: 'Participation Analysis', desc: 'Survey participation rates and completion metrics across courses' },
    { src: 'assets/pheonix/confidence.png', title: 'Confidence Analysis', desc: 'Pre/post training confidence level comparisons' },
    { src: 'assets/pheonix/statistical_analysis.png', title: 'Statistical Analysis', desc: 'Comprehensive statistical analysis with significance testing' },
    { src: 'assets/pheonix/confidence _analysis.png', title: 'Detailed Confidence Analysis', desc: 'Statistical analysis with t-tests and effect sizes' },
    { src: 'assets/pheonix/NPS.png', title: 'NPS Analysis', desc: 'Net Promoter Score analysis and trends' },
    { src: 'assets/pheonix/Sentiment_analysis.png', title: 'Sentiment Analysis', desc: 'Sentiment analysis of survey feedback and comments' }
];

let currentPhoenixIndex = 0;

function openPhoenixGallery(startIndex = 0) {
    currentPhoenixIndex = startIndex;

    if (vizModal && vizModalBody) {
        vizModalBody.innerHTML = `
            <div class="phoenix-gallery">
                <div class="phoenix-gallery-main">
                    <button class="phoenix-nav phoenix-prev" aria-label="Previous image">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M15 18l-6-6 6-6"/>
                        </svg>
                    </button>
                    <div class="phoenix-image-container">
                        <img src="${phoenixImages[currentPhoenixIndex].src}" alt="${phoenixImages[currentPhoenixIndex].title}" class="phoenix-main-image">
                    </div>
                    <button class="phoenix-nav phoenix-next" aria-label="Next image">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M9 18l6-6-6-6"/>
                        </svg>
                    </button>
                </div>
                <div class="phoenix-info">
                    <h3 class="phoenix-title">${phoenixImages[currentPhoenixIndex].title}</h3>
                    <p class="phoenix-desc">${phoenixImages[currentPhoenixIndex].desc}</p>
                    <span class="phoenix-counter">${currentPhoenixIndex + 1} / ${phoenixImages.length}</span>
                </div>
                <div class="phoenix-thumbnails">
                    ${phoenixImages.map((img, i) => `
                        <button class="phoenix-thumb ${i === currentPhoenixIndex ? 'active' : ''}" data-index="${i}">
                            <img src="${img.src}" alt="${img.title}">
                        </button>
                    `).join('')}
                </div>
            </div>
        `;

        vizModal.classList.add('active');
        document.body.style.overflow = 'hidden';

        // Add event listeners for gallery navigation
        setupPhoenixGalleryListeners();
    }
}

function setupPhoenixGalleryListeners() {
    const prevBtn = document.querySelector('.phoenix-prev');
    const nextBtn = document.querySelector('.phoenix-next');
    const thumbs = document.querySelectorAll('.phoenix-thumb');

    if (prevBtn) {
        prevBtn.addEventListener('click', () => navigatePhoenixGallery(-1));
    }

    if (nextBtn) {
        nextBtn.addEventListener('click', () => navigatePhoenixGallery(1));
    }

    thumbs.forEach(thumb => {
        thumb.addEventListener('click', () => {
            const index = parseInt(thumb.dataset.index);
            updatePhoenixGallery(index);
        });
    });

    // Keyboard navigation
    document.addEventListener('keydown', handlePhoenixKeyboard);
}

function handlePhoenixKeyboard(e) {
    if (!vizModal.classList.contains('active')) return;

    if (e.key === 'ArrowLeft') {
        navigatePhoenixGallery(-1);
    } else if (e.key === 'ArrowRight') {
        navigatePhoenixGallery(1);
    }
}

function navigatePhoenixGallery(direction) {
    let newIndex = currentPhoenixIndex + direction;
    if (newIndex < 0) newIndex = phoenixImages.length - 1;
    if (newIndex >= phoenixImages.length) newIndex = 0;
    updatePhoenixGallery(newIndex);
}

function updatePhoenixGallery(index) {
    currentPhoenixIndex = index;

    const mainImage = document.querySelector('.phoenix-main-image');
    const title = document.querySelector('.phoenix-title');
    const desc = document.querySelector('.phoenix-desc');
    const counter = document.querySelector('.phoenix-counter');
    const thumbs = document.querySelectorAll('.phoenix-thumb');

    if (mainImage) {
        mainImage.style.opacity = '0';
        setTimeout(() => {
            mainImage.src = phoenixImages[index].src;
            mainImage.alt = phoenixImages[index].title;
            mainImage.style.opacity = '1';
        }, 150);
    }

    if (title) title.textContent = phoenixImages[index].title;
    if (desc) desc.textContent = phoenixImages[index].desc;
    if (counter) counter.textContent = `${index + 1} / ${phoenixImages.length}`;

    thumbs.forEach((thumb, i) => {
        thumb.classList.toggle('active', i === index);
    });
}

// Handle Phoenix viz card click
document.addEventListener('DOMContentLoaded', () => {
    const phoenixExpandBtn = document.querySelector('.viz-card-expand[data-viz="phoenix"]');
    const phoenixCard = document.querySelector('.viz-card[data-viz="phoenix"]');

    if (phoenixExpandBtn) {
        phoenixExpandBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            openPhoenixGallery(0);
        });
    }

    if (phoenixCard) {
        phoenixCard.style.cursor = 'pointer';
        phoenixCard.addEventListener('click', (e) => {
            if (e.target.closest('.viz-card-expand')) return;
            openPhoenixGallery(0);
        });
    }
});

// ===================================
// CONSOLE MESSAGE
// ===================================

console.log('%c Shourya Thapliyal', 'font-size: 20px; font-weight: bold; color: #00D9FF;');
console.log('%cData Scientist | ML Engineer', 'font-size: 14px; color: #888;');
console.log('%cshouryat32@gmail.com', 'font-size: 12px; color: #00D9FF;');
console.log('%c Portfolio loaded', 'color: #4ADE80; font-size: 12px;');
