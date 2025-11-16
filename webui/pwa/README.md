# QuestFoundry PWA

Mobile-first Progressive Web App for QuestFoundry.

## Architecture

This PWA is **loosely coupled** to the API server and communicates via REST API only.

Key features:

- Mobile-first responsive design
- Offline capability (future)
- Hot/Cold SoT visualization
- Goal-driven workflow interface
- Gatecheck quality validation UI

## Tech Stack

- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite
- **Routing**: React Router
- **PWA**: vite-plugin-pwa
- **Styling**: TBD (Tailwind CSS recommended)

## Core Concepts Exposed in UI

1. **Projects**: User's QuestFoundry projects
2. **Hot SoT**: Working artifacts (editable, drafts)
3. **Cold SoT**: Validated artifacts (immutable, ship-ready)
4. **Goals**: Natural language input for orchestrator
5. **Gatecheck**: Quality validation workflow
6. **Provider Keys**: BYOK configuration

## Development

```bash
# Install dependencies
npm install

# Run dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Environment Variables

Create `.env.local`:

```
VITE_API_URL=http://localhost:8000
```

## Deployment

The production build is served via Nginx in Docker:

```dockerfile
FROM nginx:alpine
COPY dist /usr/share/nginx/html
```

## Implementation Status

- [ ] Project scaffolding
- [ ] Authentication flow
- [ ] Project list/create/delete
- [ ] Hot SoT viewer
- [ ] Cold SoT viewer
- [ ] Goal input interface
- [ ] Gatecheck workflow
- [ ] Provider key management
- [ ] Responsive design
- [ ] PWA manifest and service worker
