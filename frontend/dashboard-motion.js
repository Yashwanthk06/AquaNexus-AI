const motion = window.matchMedia("(hover: hover) and (prefers-reduced-motion: no-preference)");
let frame = 0;
let cursorX = 0;
let cursorY = 0;

window.addEventListener("pointermove", event => {
  if (!motion.matches || event.pointerType !== "mouse") return;

  cursorX = event.clientX;
  cursorY = event.clientY;
  if (frame) return;

  frame = requestAnimationFrame(() => {
    frame = 0;
    const root = document.documentElement;
    root.style.setProperty("--cursor-x", `${cursorX}px`);
    root.style.setProperty("--cursor-y", `${cursorY}px`);

    const hero = document.querySelector("main > header");
    if (hero) {
      const bounds = hero.getBoundingClientRect();
      const x = Math.max(0, Math.min(bounds.width, cursorX - bounds.left));
      const y = Math.max(0, Math.min(bounds.height, cursorY - bounds.top));
      hero.style.setProperty("--hero-x", `${x}px`);
      hero.style.setProperty("--hero-y", `${y}px`);
    }
  });
}, { passive: true });
