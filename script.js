// script.js

document.addEventListener('DOMContentLoaded', (event) => {
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();

            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Simple responsive menu toggle
    const navToggle = document.createElement('button');
    navToggle.textContent = 'Menu';
    navToggle.classList.add('nav-toggle');
    document.querySelector('header .container').prepend(navToggle);

    navToggle.addEventListener('click', () => {
        document.querySelector('nav').classList.toggle('show');
    });

    // Header scroll effect
    const header = document.querySelector('header');
    let lastScrollTop = 0;

    window.addEventListener('scroll', () => {
        let scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        if (scrollTop > lastScrollTop) {
            header.style.top = '-80px';
        } else {
            header.style.top = '0';
        }
        lastScrollTop = scrollTop;
    });

    // Add animation class to products when they come into view
    const products = document.querySelectorAll('.product');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animate');
            }
        });
    }, { threshold: 0.1 });

    products.forEach(product => {
        observer.observe(product);
    });
});