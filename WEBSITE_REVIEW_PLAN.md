# AUSDAIS Website Review and Improvement Plan

This document outlines the plan for reviewing and improving the AUSDAIS website based on an analysis of the local project files.

**Key Requirements:**

*   Ensure the correct domain `ausdais.au` is used consistently.
*   Remove any explicit street addresses or phone numbers from the website content and code.

## Phase 1: Issue Remediation

1.  **Fix Broken Links:**
    *   Remove or update the links to `industries.html` and `resources.html` in the footer across all relevant HTML files (`index.html`, `about.html`, etc.).
2.  **Correct Inconsistencies:**
    *   Standardize the logo used in headers/footers across all pages (decide between `logogreen.png` or `logoblue.png`).
    *   Standardize the main navigation across all pages (decide if "Privacy Policy" should be on all pages).
    *   Ensure the correct contact email address (`info@ausdais.au`) is used consistently in footers and contact sections, replacing any instances of `info@ausdais.com`.
3.  **Fix Contact Form & Details:**
    *   Determine the intended functionality for the contact form (currently pointing to non-existent `submit-form.php`). Options:
        *   Implement a server-side script (e.g., using `app.py`).
        *   Use a third-party form service.
        *   Replace the form with a simple `mailto:info@ausdais.au` link.
    *   **Crucially:** Remove any commented-out or visible street addresses and phone numbers from the HTML (e.g., in `index.html` contact section). Ensure only the official email (`info@ausdais.au`) is present if contact details are displayed directly.
4.  **Address JavaScript/CSS Issues:**
    *   Add the necessary CSS rules for `.nav-toggle` and `nav.show` to make the mobile menu functional.
    *   Define the `.animate` CSS class with appropriate animation properties for the product section.
    *   Consider making dropdown menus activate on click for better touch device compatibility.
5.  **Clean Up HTML:**
    *   Remove commented-out sections (Testimonials, Login) if they are not planned for implementation soon.
    *   Move inline styles (like `color: white;` on line 84 of `index.html`) to `style.css`.

## Phase 2: Enhancements & Best Practices

6.  **Improve Accessibility:**
    *   Review and update image `alt` text to be more descriptive across all pages.
    *   Check color contrast ratios to ensure readability.
    *   Ensure interactive elements have clear focus indicators.
7.  **Enhance Responsiveness:**
    *   Thoroughly test the layout on various screen sizes (mobile, tablet, desktop).
    *   Refine CSS media queries as needed to fix any layout breaks or awkward spacing.
8.  **Content Review:**
    *   Review placeholder text (especially on `about.html`) and update with specific, accurate information about AusDAIS.
    *   Proofread all content for grammatical errors and typos.
9.  **Code Quality:**
    *   Review CSS for potential redundancy or simplification opportunities.
    *   Ensure consistent code formatting.

## Phase 3: Final Checks

10. **Cross-Browser Testing:** Test the site on major web browsers (Chrome, Firefox, Safari, Edge).
11. **Validation:** Validate HTML and CSS using online validators.

## Workflow Overview

```mermaid
graph TD
    A[Start Review] --> B{Analyze Files};
    B --> C[Identify Issues & Inconsistencies];
    C --> D[Phase 1: Fixes];
    D --> D1[Broken Links];
    D --> D2[Inconsistencies (Logo, Nav, Email)];
    D --> D3[Contact Form/Details (Remove Address/Phone)];
    D --> D4[JS/CSS Bugs (Mobile Menu, Animation)];
    D --> D5[HTML Cleanup];
    D5 --> E[Phase 2: Enhancements];
    E --> E1[Accessibility (Alt Text, Contrast)];
    E --> E2[Responsiveness];
    E --> E3[Content Update];
    E --> E4[Code Quality];
    E4 --> F[Phase 3: Final Checks];
    F --> F1[Cross-Browser Test];
    F --> F2[Validation];
    F2 --> G[Review Complete];