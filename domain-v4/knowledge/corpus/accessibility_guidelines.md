# Accessibility Guidelines for Interactive Fiction

Craft guidance for making IF accessible to diverse readers—cognitive accessibility, visual considerations, input options, and inclusive design.

---

## Why Accessibility Matters

### The Business Case

- 15-20% of population has some disability
- Accessible design benefits everyone
- Legal requirements in many jurisdictions
- Larger audience = more readers

### The Ethical Case

- Stories should be available to all
- Exclusion by design is a choice
- Disability is part of human diversity
- Good design is inclusive design

### The Design Case

- Constraints drive creativity
- Accessible solutions often cleaner
- Benefits extend beyond target users
- "Curb cut effect"—helps everyone

---

## Cognitive Accessibility

### Reading Level Considerations

**Factors Affecting Comprehension:**

- Vocabulary complexity
- Sentence length and structure
- Information density
- Abstract vs. concrete language

**Adjustable Options:**

- Text complexity settings
- Vocabulary tooltips
- Simplified mode available
- Content summaries

### Memory Demands

**Challenges:**

- Tracking multiple plot threads
- Remembering earlier choices
- Understanding cause-effect across time
- Character and place names

**Solutions:**

- Story journal/recap features
- Character/location reference
- "Previously on" summaries
- Visual relationship maps

### Decision Fatigue

**Challenges:**

- Too many choices exhaust
- Complex consequences overwhelm
- Time pressure adds stress

**Solutions:**

- Limit choices per scene (2-4 typical)
- Clear consequence hints when helpful
- Optional "story mode" with reduced choices
- Allow choice review before confirming

### Navigation and Orientation

**Challenges:**

- Getting lost in branching stories
- Forgetting where you are
- Losing track of goals

**Solutions:**

- Clear scene/chapter indicators
- Quest log or objective tracker
- Map of story progress
- "Where am I?" feature

---

## Visual Accessibility

### Text Readability

**Font Considerations:**

- Readable typeface (avoid decorative fonts for body)
- Sufficient size (adjustable preferred)
- Clear letter differentiation (l/1/I, O/0)
- Adequate line spacing

**User Controls:**

- Font size adjustment
- Font family options (including dyslexia-friendly)
- Line spacing adjustment
- Margin/padding controls

### Contrast and Color

**Minimum Contrast:**

- 4.5:1 for normal text
- 3:1 for large text
- Check with contrast checking tools

**Color Independence:**

- Don't convey meaning through color alone
- Use icons, patterns, or labels alongside color
- Test with color blindness simulators

**Theme Options:**

- Light and dark modes
- High contrast option
- Custom color schemes

### Screen Reader Compatibility

**Requirements:**

- Proper heading structure (H1, H2, H3)
- Alt text for meaningful images
- Form labels for inputs
- Logical reading order
- Skip navigation options

**Testing:**

- Test with actual screen readers
- Verify all content accessible
- Check that choices are navigable
- Ensure state changes announced

---

## Motor Accessibility

### Input Methods

**Support Multiple Input Types:**

- Mouse/touch
- Keyboard only
- Voice commands (where available)
- Switch devices
- Eye tracking (specialized)

### Keyboard Navigation

**Requirements:**

- All functions accessible via keyboard
- Visible focus indicators
- Logical tab order
- Keyboard shortcuts for common actions
- No keyboard traps

**Best Practices:**

- Arrow keys for menu navigation
- Enter/Space for selection
- Escape to close/back
- Tab for focus movement

### Timing Considerations

**Challenges:**

- Timed choices create pressure
- Quick reactions required
- Holding buttons difficult

**Solutions:**

- Avoid mandatory time limits
- Adjustable timer speeds
- Option to disable timers
- No penalties for slow response

### Click/Tap Targets

**Requirements:**

- Minimum 44x44 pixels (touch)
- Adequate spacing between targets
- No precision clicking required
- Clear visual boundaries

---

## Audio Accessibility

### For Deaf/Hard of Hearing

**Captions/Subtitles:**

- Full text for all audio content
- Speaker identification
- Sound effect descriptions [door creaks]
- Music mood indicators [tense music]

**Visual Alternatives:**

- No audio-only information
- Visual cues for important sounds
- Vibration options where applicable

### For Blind/Low Vision

**Audio Description:**

- Describe visual elements
- Narrate scene context
- Text-to-speech for all content
- Sound design that conveys information

**Screen Reader Support:**

- See Visual Accessibility section
- Ensure all UI elements labeled
- Announce state changes

---

## Content Accessibility

### Content Warnings

**Why Warnings Matter:**

- Allow informed consent
- Prevent trauma triggers
- Respect reader autonomy
- Build trust

**Implementation:**

- Upfront content advisory
- Specific warnings (not vague "mature content")
- Option to skip or minimize triggering content
- Non-spoilery where possible

### Sensitive Content Handling

**Options to Consider:**

- Content filters (violence, romance, etc.)
- Reduced intensity modes
- Text-only descriptions vs. graphic detail
- Skip scene functionality

### Cultural Accessibility

**Considerations:**

- Date/time format options
- Currency/measurement localization
- Cultural references explained
- Avoid assumptions about reader background

---

## Customization Framework

### Settings to Offer

**Text Display:**

- Font size
- Font family
- Line spacing
- Text color
- Background color
- Contrast mode

**Audio:**

- Volume controls (separate for music/effects/voice)
- Captions on/off
- Audio description on/off
- Screen reader optimization

**Input:**

- Control remapping
- Timer settings
- One-hand mode
- Switch access support

**Gameplay:**

- Difficulty/assistance levels
- Content filters
- Hint frequency
- Auto-save frequency

### Preset Profiles

Consider offering preset accessibility profiles:

- Dyslexia-friendly (font, spacing, colors)
- Low vision (large text, high contrast)
- Motor accessible (keyboard focus, no timers)
- Cognitive support (hints, recaps, simplified)

---

## Testing for Accessibility

### Automated Testing

**Tools:**

- WAVE (web accessibility evaluation)
- axe (automated testing)
- Lighthouse (Chrome DevTools)
- Color contrast checkers

**Limitations:**

- Catches ~30% of issues
- Can't evaluate usability
- False positives possible
- Must combine with manual testing

### Manual Testing

**Keyboard Testing:**

- Navigate entire IF with keyboard only
- Check focus visibility
- Verify all functions accessible
- Test without mouse connected

**Screen Reader Testing:**

- Test with NVDA, JAWS, or VoiceOver
- Verify all content read correctly
- Check navigation and orientation
- Test with actual users if possible

### User Testing

**Most Valuable:**

- Test with disabled users
- Diverse disabilities, diverse solutions
- Observe actual usage
- Listen to feedback

**Where to Find Testers:**

- Accessibility communities
- Beta reader calls specifying accessibility
- Disability organizations
- Accessibility consultants

---

## Implementation Priorities

### Start Here (High Impact, Lower Effort)

1. Proper heading structure
2. Sufficient color contrast
3. Keyboard navigation
4. Text size adjustment
5. Content warnings

### Next Steps (Medium Effort)

1. Screen reader optimization
2. Font options
3. Dark/light modes
4. Caption support
5. Choice assistance options

### Advanced (Higher Effort)

1. Full customization framework
2. Multiple input methods
3. Audio description
4. Cognitive support systems
5. User testing program

---

## Common Mistakes

### Accessibility as Afterthought

Adding accessibility at the end is expensive and incomplete.

**Fix:** Design with accessibility from the start.

### Assuming You Know What Users Need

Guessing at accessibility needs without consulting disabled users.

**Fix:** Involve disabled users in design and testing.

### Override User Settings

Ignoring system accessibility settings (font size, colors).

**Fix:** Respect and respond to system settings.

### Accessible Means Ugly

Believing accessible design must sacrifice aesthetics.

**Fix:** Good design can be both beautiful and accessible.

### One Size Fits All

Single "accessibility mode" treating all disabilities same.

**Fix:** Multiple options for different needs.

### Test With Tools Only

Relying solely on automated testing.

**Fix:** Combine automated, manual, and user testing.

---

## Quick Reference

| Category | Key Actions |
|----------|-------------|
| Cognitive | Recaps, journals, clear navigation, limited choices |
| Visual | Contrast, font options, screen reader support, no color-only info |
| Motor | Keyboard access, large targets, adjustable timing, multiple inputs |
| Audio | Captions, visual alternatives, TTS, audio description |
| Content | Warnings, filters, skip options, cultural awareness |
| Testing | Automated + manual + user testing |
| Priority | Start with structure, contrast, keyboard, content warnings |
