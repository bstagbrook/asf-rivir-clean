# Shapester Learning Guide

## How to Get REALLY Good Software

The description IS the software. More detail = better results.

### The Secret

```
Quality(output) = f(Quality(description))
```

The shapester learns from satisfaction signals. Every time you rate something, it remembers which features correlated with high scores.

### Description Levels

**Level 1: Basic (Weak)**
```
"make a todo app"
```
Result: Generic todo app with default styling.

**Level 2: Better**
```
"create a calm todo list with soft colors"
```
Result: Todo app with calm mood, softer color scheme.

**Level 3: Good**
```
"create a calm, elegant todo list with soft lavender colors and smooth animations"
```
Result: Elegant todo with purple scheme, refined typography, smooth transitions.

**Level 4: Badass**
```
"craft a mindful task tracker with warm earth tones, gentle animations,
elegant serif typography, rounded corners, subtle shadows, and a sacred,
meditative mood that inspires focused productivity"
```
Result: Stunning, production-quality app tailored to your exact vision.

### Feature Vocabulary

The shapester recognizes these features:

**Aesthetic**
- minimal, minimalist, simple, clean, bare
- elegant, refined, sophisticated, graceful
- bold, striking, dramatic, powerful
- soft, gentle, subtle, quiet
- modern, contemporary, sleek, fresh
- classic, traditional, timeless, vintage
- playful, fun, whimsical, cheerful
- professional, corporate, business, serious

**Mood**
- calm, peaceful, serene, tranquil, zen
- energetic, vibrant, lively, dynamic
- focused, concentrated, productive, efficient
- creative, artistic, expressive, inspired
- cozy, warm, comfortable, inviting
- sacred, spiritual, mindful, meditative

**Colors**
- dark, black, night, shadow
- light, white, bright, airy
- warm, golden, amber, orange, yellow, sunset
- cool, blue, teal, cyan, ocean
- earth, brown, beige, natural, organic
- purple, violet, lavender, indigo
- green, forest, mint, sage, emerald
- pink, rose, coral, blush

**Animations**
- smooth, fluid, seamless, flowing
- bounce, spring, elastic
- fade, dissolve, transition
- slide, glide, sweep
- grow, expand, scale, zoom
- static, instant, no animation

**Layout**
- grid, tile, mosaic, card
- list, stack, column, vertical
- split, sidebar, panel, dual
- centered, focused, spotlight
- fullscreen, immersive, full

### Maximizing Learning

1. **Rate Honestly** - The shapester learns from your real preferences
2. **Provide Feedback** - Qualitative feedback helps explain the score
3. **Be Specific** - "I love the color but hate the font" > "it's okay"
4. **Iterate** - Try variations to help the system learn distinctions
5. **Use Rich Vocabulary** - More words = more features to learn from

### Training Loop

```
1. Describe what you want (in detail!)
2. See what the shapester generates
3. Rate it (0-10)
4. Provide feedback
5. Repeat
```

Each iteration:
- Reinforces good features
- De-prioritizes bad features
- Builds your personal preference model

### Commands

```bash
# Interactive session
python3 badass_shapester.py

# See your best features
> best

# Get suggestions for a description
> craft a todo app
Suggestion: craft a todo app with calm, elegant, minimal aesthetic
Use improved? (y/n):
```

### Example Session

```
> craft a mindful daily journal with warm earth tones, elegant serif
  typography, peaceful mood, and subtle animations

  Detected features:
    Target:     notes
    Aesthetic:  elegant
    Mood:       calm
    Colors:     earth
    Detail:     ★★★★☆☆☆☆☆☆

  Generated: badass_notes.html
  Open in browser to experience it!

  Rate it (0-10): 9
  Feedback: love the typography, colors are perfect

  Learned! Satisfaction: 90%
  Your top features: elegant, earth, calm
```

### Integration with Shape OS

The shapester integrates with:
- **Structural Intelligence** - Satisfaction invariants
- **Receipt Store** - Full audit trail
- **Double Membrane** - Recognition filtering
- **Waveform Model** - Clockless execution

Every generation creates a receipt. Every rating trains the invariants.

### The Goal

Over time, the shapester should:
1. Know YOUR preferences exactly
2. Suggest improvements automatically
3. Generate production-quality on first try
4. Create software that delights YOU specifically

The description IS the software.
The rating IS the training.
The delight IS the goal.
