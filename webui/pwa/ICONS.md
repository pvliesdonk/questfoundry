# PWA Icon Generation Guide

## Quick Start

Generate PNG icons from the SVG source:

```bash
# Install sharp (if not already installed)
npm install --save-dev sharp

# Generate icons
node scripts/generate-icons.js
```

## Icon Requirements

### PWA Manifest Icons

- **192x192px**: Minimum required size for PWA
- **512x512px**: Recommended for splash screens

### Apple Touch Icons

- **180x180px**: For iOS home screen

### Favicons

- **32x32px**: Standard favicon
- **16x16px**: Small favicon
- **favicon.ico**: Multi-size ICO file

## Current Status

The repository includes:

- ✅ `icon.svg` - Source SVG icon (QF branding)
- ⚠️ PNG placeholders - Need to be generated

## Manual Icon Generation

If you can't use the script, generate icons manually:

### Option 1: Online Tools

1. Visit <https://realfavicongenerator.net/>
2. Upload `/public/icon.svg`
3. Download generated package
4. Replace files in `/public/`

### Option 2: ImageMagick

```bash
# Install ImageMagick
sudo apt-get install imagemagick

# Generate icons
convert public/icon.svg -resize 192x192 public/icon-192x192.png
convert public/icon.svg -resize 512x512 public/icon-512x512.png
convert public/icon.svg -resize 180x180 public/apple-touch-icon.png
convert public/icon.svg -resize 32x32 public/favicon-32x32.png
convert public/icon.svg -resize 16x16 public/favicon-16x16.png
```

### Option 3: Inkscape

```bash
# Install Inkscape
sudo apt-get install inkscape

# Generate icons
inkscape -w 192 -h 192 public/icon.svg -o public/icon-192x192.png
inkscape -w 512 -h 512 public/icon.svg -o public/icon-512x512.png
inkscape -w 180 -h 180 public/icon.svg -o public/apple-touch-icon.png
```

## Custom Icon Design

To create a custom icon:

1. **Edit** `/public/icon.svg` with your design
2. **Requirements**:
   - Square aspect ratio (viewBox="0 0 512 512")
   - Visible at small sizes (192x192)
   - Distinctive at large sizes (512x512)
   - Works on light and dark backgrounds
3. **Generate** PNGs using the script
4. **Test** on mobile devices

## Icon Design Tips

- ✅ Use simple, bold shapes
- ✅ High contrast colors
- ✅ Readable at 48x48 (Android)
- ✅ Looks good on both light/dark backgrounds
- ❌ Avoid fine details
- ❌ Avoid text smaller than 20px
- ❌ Avoid gradients (can look poor at small sizes)

## Testing

After generating icons:

1. **Build**: `npm run build`
2. **Serve**: `npm run preview`
3. **Test PWA**:
   - Chrome DevTools > Application > Manifest
   - Check icon displays correctly
   - Try "Install app" prompt
4. **Mobile Test**:
   - Deploy to test server
   - Add to home screen on iOS/Android
   - Verify icons appear correctly

## Troubleshooting

**Icons not updating?**

- Clear browser cache (Ctrl+Shift+R)
- Clear service worker cache
- Check browser console for errors

**Icons look blurry?**

- Ensure PNG is high resolution
- SVG should be vector (not rasterized)
- Check PNG export settings (no compression artifacts)

**PWA not installable?**

- Check manifest.json has icons array
- Verify icons are accessible (not 404)
- Ensure HTTPS (required for PWA)

## Resources

- [Web App Manifest Spec](https://www.w3.org/TR/appmanifest/)
- [PWA Icon Guidelines](https://web.dev/add-manifest/)
- [Favicon Best Practices](https://github.com/audreyfeldroy/favicon-cheat-sheet)
