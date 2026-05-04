export function initNavbar() {

    window.addEventListener("scroll", () => {
        const navbar = document.querySelector(".navbar");
        if (!navbar) return;

        navbar.classList.toggle("scrolled", window.scrollY > 40);
    });

}