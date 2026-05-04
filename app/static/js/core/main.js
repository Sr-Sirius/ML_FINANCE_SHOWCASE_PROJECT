import { initGlow } from "./glow.js";
import { initReveal } from "./reveal.js";
import { initNavbar } from "./navbar.js";

document.addEventListener("DOMContentLoaded", () => {
    initGlow();
    initReveal();
    initNavbar();
});