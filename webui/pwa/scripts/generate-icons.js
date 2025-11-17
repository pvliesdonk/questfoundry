#!/usr/bin/env node
/**
 * Icon Generation Script for QuestFoundry PWA
 *
 * This script generates PNG icons from the SVG source icon.
 * Requires: sharp (npm install --save-dev sharp)
 *
 * Usage: node scripts/generate-icons.js
 */

const fs = require('node:fs');
const path = require('node:path');

// Check if sharp is installed
let sharp;
try {
  sharp = require('sharp');
} catch (e) {
  console.error('❌ Error: sharp is not installed');
  console.error('Please install it with: npm install --save-dev sharp');
  process.exit(1);
}

const svgPath = path.join(__dirname, '../public/icon.svg');
const publicDir = path.join(__dirname, '../public');

// Icon sizes to generate
const sizes = [
  { size: 192, name: 'icon-192x192.png', desc: 'PWA icon (small)' },
  { size: 512, name: 'icon-512x512.png', desc: 'PWA icon (large)' },
  { size: 180, name: 'apple-touch-icon.png', desc: 'Apple touch icon' },
  { size: 32, name: 'favicon-32x32.png', desc: 'Favicon' },
  { size: 16, name: 'favicon-16x16.png', desc: 'Favicon (small)' },
];

async function generateIcons() {
  console.log('🎨 Generating PWA icons from SVG...\n');

  // Read SVG file
  if (!fs.existsSync(svgPath)) {
    console.error(`❌ SVG file not found: ${svgPath}`);
    process.exit(1);
  }

  const svgBuffer = fs.readFileSync(svgPath);

  // Generate each size
  for (const { size, name, desc } of sizes) {
    try {
      const outputPath = path.join(publicDir, name);
      await sharp(svgBuffer)
        .resize(size, size)
        .png()
        .toFile(outputPath);
      console.log(`✓ Generated ${name} (${size}x${size}) - ${desc}`);
    } catch (error) {
      console.error(`❌ Failed to generate ${name}:`, error.message);
    }
  }

  // Generate favicon.ico (multi-size ICO)
  console.log('\n📦 Generating favicon.ico...');
  try {
    const favicon16 = await sharp(svgBuffer).resize(16, 16).png().toBuffer();
    const favicon32 = await sharp(svgBuffer).resize(32, 32).png().toBuffer();

    // Note: Creating a proper .ico requires additional libraries
    // For now, we'll just create a 32x32 PNG as favicon.ico
    await sharp(favicon32).toFile(path.join(publicDir, 'favicon.ico'));
    console.log('✓ Generated favicon.ico (use ico-converter for true .ico format)');
  } catch (error) {
    console.error('❌ Failed to generate favicon.ico:', error.message);
  }

  console.log('\n✨ Icon generation complete!\n');
  console.log('Generated icons:');
  for (const { name } of sizes) {
    console.log(`  - ${name}`);
  }
  console.log('\nNext steps:');
  console.log('  1. Review generated icons in /public/');
  console.log('  2. Run: npm run build');
  console.log('  3. Test PWA installation');
}

generateIcons().catch(error => {
  console.error('❌ Icon generation failed:', error);
  process.exit(1);
});
