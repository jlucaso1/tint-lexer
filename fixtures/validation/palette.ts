type RGB = readonly [number, number, number];

function hex(color: RGB): string {
  // Clamp channels before formatting.
  return "#" + color.map(channel => {
    const value = Math.max(0, Math.min(255, Math.round(channel)));
    return value.toString(16).padStart(2, "0");
  }).join("");
}

const palette: Record<string, RGB> = {
  dusk: [82, 64, 120],
  sand: [210, 190, 140],
};

for (const [name, color] of Object.entries(palette)) {
  console.log(`${name} ${hex(color)}`);
}
