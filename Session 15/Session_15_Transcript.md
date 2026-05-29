# Session 15 — Video Analytics I
### Lecture Transcript
**Topics:** Optical Flow · Background Subtraction · Motion Detection
**Format:** Full Theory Session · ~1 Hour

---

## Opening

Welcome to Session 15. Today we're going to cover a topic that sits right at the heart of video understanding — not just looking at individual images, but actually understanding *how things move* across time.

We'll cover three closely related techniques: **Optical Flow**, **Background Subtraction**, and **Motion Detection**. By the end of this session you'll understand how each one works, why they were designed the way they are, and how they fit together in a real video analytics pipeline.

Let's look at the roadmap first so you know where we're headed.

We have three parts. Part 1 is Optical Flow — this is the most mathematically rich section, covering how we compute per-pixel motion vectors between frames. Part 2 is Background Subtraction — a more practical, statistics-based approach to separating moving objects from a static background. And Part 3 is Motion Detection — which is really the output stage, where we take what background subtraction gives us and turn it into actual detected objects with locations and sizes.

---

## Part 1 — Optical Flow

### What is Optical Flow?

Let's start with the fundamental question: what is optical flow?

Optical flow is the *pattern of apparent motion* of objects, surfaces, and edges in a visual scene, caused by relative motion between a camera and the scene. In plain terms — when something moves in front of a camera, or the camera itself moves, the pixels in the image shift. Optical flow is the mathematical description of exactly *how* and *where* those pixels shifted.

The output of optical flow is a **2D vector field**. For every pixel in the image — or for a selected set of pixels — you get a vector `(u, v)` that tells you: this pixel moved `u` pixels to the right and `v` pixels downward between frame `t` and frame `t+1`.

Think of it like this: imagine a ball rolling across a table, captured in a video. For each frame pair, optical flow draws an arrow on every pixel of that ball showing exactly how far and in which direction it moved. The arrows together form a dense "flow field" over the image.

There are two flavors: **sparse** optical flow, where you only compute vectors for a few carefully chosen feature points, and **dense** optical flow, where you compute a vector for every single pixel. We'll look at both.

---

### The Two Core Assumptions

Optical flow rests on two assumptions that are worth understanding deeply, because they directly determine where the technique works well and where it fails.

**Assumption 1: Brightness Constancy**

This says that a pixel's intensity does not change between consecutive frames, even if it moves to a new location. Mathematically:

```
I(x, y, t)  =  I(x+dx, y+dy, t+dt)
```

The intensity at position `(x, y)` at time `t` equals the intensity at the new position `(x+dx, y+dy)` at time `t+dt`. This is also called the *Optical Flow Constraint*.

This assumption is violated whenever lighting changes — if a light flickers, or a cloud passes over the sun. That's a known weakness.

**Assumption 2: Spatial Coherence**

This says that neighboring pixels belong to the same surface and therefore move together. They share the same velocity vector `(u, v)`.

This assumption is what allows us to actually *solve* for the motion — without it, the problem is mathematically under-determined. We'll see exactly why in a moment.

---

### The Optical Flow Equation

Here's where the math comes together. If we take the brightness constancy equation and apply a **Taylor series expansion** around the current position, then divide through by `dt`, we arrive at the fundamental optical flow equation:

```
Ix * u  +  Iy * v  +  It  =  0
```

Let's break down every term:

- `Ix` and `Iy` are the **spatial image gradients** — how quickly intensity changes in the x and y directions. These are computed directly from the image using standard derivative filters.
- `It` is the **temporal gradient** — how much the intensity at a given pixel changed between the two frames.
- `u` and `v` are the **velocity components** we want to find — the horizontal and vertical flow at this pixel.

`Ix`, `Iy`, and `It` are all known — we can compute them from the images. But `u` and `v` are unknown.

Here's the problem: **one equation, two unknowns**. The system is under-determined. For any given pixel, there are infinitely many `(u, v)` pairs that satisfy this equation. This is the famous **Aperture Problem**.

---

### The Aperture Problem

Imagine watching a striped bar move through a small circular hole — an aperture. You can tell the bar is moving, but you cannot determine *which direction* it's actually moving. It could be moving left, or diagonally, or at many other angles — and through the small window they'd all look the same. You can only measure the component of motion *perpendicular* to the stripe direction.

This is exactly the mathematical problem we just described: one equation, two unknowns means you can only constrain motion in one direction per pixel.

How do different methods solve this?

- **Lucas-Kanade** adds the spatial coherence assumption: it takes a small patch of neighboring pixels and assumes they all share the same velocity. With a 3×3 patch that gives 9 equations for 2 unknowns — now massively over-determined, solved by least squares.
- **Horn-Schunck** uses global smoothness regularization — it adds a penalty for the flow field varying too quickly across the image.
- **Deep Learning methods** like RAFT and FlowNet learn the right constraints directly from training data with known ground-truth flow.

---

### Sparse vs Dense Optical Flow

The two main practical approaches differ in *how many pixels* they track.

**Sparse Optical Flow** selects a set of "good" feature points — typically corners or highly textured regions — and tracks only those between frames. The key algorithm here is **Lucas-Kanade**, which we'll look at next. The advantages are speed and robustness to noise. The limitation is that you only have motion information at a sparse set of points, not across the whole image.

**Dense Optical Flow** computes a motion vector for every single pixel. The key algorithm is **Farneback**. The advantages are a complete motion field and rich visualization. The limitation is that it's significantly more computationally expensive.

In OpenCV:
- Sparse: `cv2.calcOpticalFlowPyrLK()`
- Dense: `cv2.calcOpticalFlowFarneback()`

---

### Lucas-Kanade: Sparse Optical Flow

Lucas-Kanade is the classic sparse optical flow algorithm. The core idea is to directly address the aperture problem using the spatial coherence assumption.

Here's the step-by-step:

**Step 1 — Detect corners.** Use Shi-Tomasi or Harris corner detection to find "good features to track." Corners are ideal because they have strong gradients in two directions, giving a well-conditioned system of equations.

**Step 2 — Apply the patch constraint.** For each feature point, take a small 3×3 patch centered on it. Each of the 9 pixels gives one instance of the optical flow equation `Ix*u + Iy*v + It = 0`. Now we have 9 equations with 2 unknowns — an over-determined system.

**Step 3 — Solve by least squares.** Write this as a matrix equation `Av = b` and solve for the optimal `(u, v)` that best satisfies all 9 constraints simultaneously. This gives a unique, robust solution.

**Step 4 — Image pyramid.** For large displacements between frames, Lucas-Kanade would fail because the brightness constancy assumption only holds for small motions. The solution is a multi-scale pyramid: compute flow at the coarsest level first, then refine progressively at finer scales. This is why it's called *Pyramidal* Lucas-Kanade.

The result you see from this algorithm is a set of tracked points with colored trails showing their trajectories over time — great for tracking people, vehicles, or feature points in a scene.

---

### Farneback: Dense Optical Flow

Farneback's algorithm takes a completely different approach to produce a dense flow field — one motion vector per pixel.

The key idea is **polynomial expansion**: instead of working directly with pixel intensities, the algorithm approximates the neighborhood of each pixel as a polynomial function. Then it uses the displacement of these polynomial approximations between frames to estimate motion.

The output of Farneback is a full 2D flow field — a two-channel image where one channel contains the horizontal velocities and the other contains the vertical velocities for every pixel.

To *visualize* this, we convert it to **HSV color space**:
- **Hue** encodes the direction of motion — different directions get different colors
- **Saturation** is fixed at maximum (255)
- **Value** encodes the *magnitude* of motion — fast-moving pixels are bright, stationary ones are dark

This gives you the characteristic colorful dense flow images you've probably seen — blobs of color indicating moving objects against a black background where nothing is moving.

The key difference from Lucas-Kanade: there is no feature selection step. The entire frame is processed. Every pixel gets a flow vector.

---

### Real-World Applications of Optical Flow

Optical flow shows up in a surprising number of places:

**Video Stabilization** — Camera shake produces a flow field that's roughly uniform across the frame (everything moving in the same direction). Estimating and inverting this field compensates for the shake.

**Autonomous Driving** — Self-driving vehicles use optical flow to distinguish between stationary background and moving objects like pedestrians and other cars, even before any object detection is applied.

**Robot Navigation** — Visual odometry uses the flow field to estimate how much the robot itself has moved — purely from pixel motion, without GPS or wheel encoders.

**Action Recognition** — Two-stream neural networks, a foundational architecture in video understanding, take optical flow as a second input stream alongside RGB frames, giving the network explicit motion information.

**Video Compression** — MPEG and H.264 use "motion vectors" between frames that are conceptually very close to optical flow. Instead of encoding every frame from scratch, you encode just the motion and the residual.

**AR & Object Tracking** — Once an object is detected, optical flow can track it across subsequent frames without re-running detection every frame, making tracking much more efficient.

---

## Part 2 — Background Subtraction

### What is Background Subtraction?

Background subtraction addresses a specific but very common problem: you have a fixed camera watching a scene, and you want to find anything that's *moving* in that scene.

The approach is conceptually elegant: build a model of what the *background* looks like — the road, the floor, the wall — and then any pixel that deviates significantly from that model must be a moving foreground object.

The output is a **binary foreground mask**: white pixels are foreground (moving), black pixels are background (static).

The process has two phases:

**Phase 1 — Background Initialization.** Look at several frames of the scene, ideally when nothing interesting is happening, to build an initial statistical model of each pixel's "normal" appearance.

**Phase 2 — Background Update.** Continuously update the model as the background slowly changes — lighting shifts across the day, shadows move, a new object is placed in the scene. The model must adapt, but slowly, so that genuinely moving objects are still detected.

---

### Simple Frame Differencing

The simplest possible approach: just subtract a reference background image from the current frame.

```
Foreground(x,y,t)  =  |I(x,y,t) - B(x,y)|  >  Threshold
```

If the absolute difference at a pixel exceeds a threshold, classify it as foreground.

This works well when:
- The camera is perfectly static
- Lighting is controlled and consistent
- Objects in the foreground are clearly different from the background

It fails badly when:
- Lighting changes — a sudden change makes the entire frame appear as foreground
- Objects stop moving — they get absorbed into the background reference and disappear
- The background itself has dynamic elements — waving trees, rippling water
- Shadows — a shadow changes pixel intensity but isn't the object itself

Frame differencing is rarely used in production systems for exactly these reasons, but it's a valuable baseline to understand before looking at more sophisticated models.

---

### Statistical Background Modeling

The better approach is to model each pixel not as a fixed value but as a **probability distribution**. The background at pixel `(x, y)` isn't one specific intensity — it's a distribution of possible intensities given all the normal variations.

**Running Average** — The simplest statistical model. The background estimate is a weighted average of all recent frames:
```
B(t) = alpha * I(t) + (1 - alpha) * B(t-1)
```
`alpha` is the learning rate — small `alpha` means the background adapts slowly, large `alpha` means it adapts quickly. Simple and fast, but treats every pixel as having a single "normal" state.

**Single Gaussian** — Each pixel is modeled as a Gaussian distribution with a mean `mu` and variance `sigma²` estimated from recent frames. A pixel is foreground if its current intensity is more than `k` standard deviations from the mean. Better than running average, but still assumes each pixel has exactly one "normal" appearance.

**Mixture of Gaussians (MOG)** — The most powerful statistical model. Each pixel is modeled as a *mixture* of K Gaussian distributions. This handles **multi-modal** backgrounds — a pixel that shows both the wall behind a window and the sky through the window can have two valid "normal" states. MOG adapts to these situations naturally. This is the foundation of the algorithms used in practice.

---

### MOG2: OpenCV's Background Subtractor

**MOG2** (Mixture of Gaussians, version 2) by Zivkovic (2004) is the standard background subtractor in OpenCV. It improves on the basic MOG in several important ways:

**Adaptive K** — Basic MOG requires you to specify the number of Gaussians per pixel in advance. MOG2 automatically adjusts this number per pixel, from 1 to 5, based on the complexity of that pixel's history. Simple static background pixels use one Gaussian; dynamic pixels (like water reflections) automatically get more.

**Shadow Detection** — MOG2 includes built-in shadow detection. Shadow pixels are marked in gray in the output mask rather than white, so downstream processing can distinguish true foreground objects from their shadows. This significantly reduces false positives in surveillance scenarios.

**Online Learning** — The background model updates every frame using a configurable history length. The `history` parameter controls how many recent frames are used to estimate the background.

**High Accuracy** — MOG2 handles illumination changes, gradual scene shifts, and repetitive motion (like a fan spinning in the background) that would confuse simpler methods.

```python
cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
```

OpenCV also includes a **KNN** (K-Nearest Neighbours) background subtractor as an alternative. KNN models the background using samples of recent pixel values rather than Gaussian distributions — it tends to be more accurate on complex backgrounds but slightly slower.

---

### MOG2 vs KNN: Visual Results

When you run both algorithms on the same traffic sequence, the output masks are broadly similar — both produce white blobs for vehicles moving through the frame. The key differences are in the details:

- The **input frame** shows a road with moving vehicles
- The **MOG2 result** shows white blobs for vehicles, with gray regions indicating shadows — the built-in shadow detection is doing its job
- The **KNN result** shows similar vehicle blobs, but the shadow handling may differ slightly in edge cases

In practice, MOG2 is the default choice for most applications. KNN is worth trying when you have particularly complex or multi-modal backgrounds.

---

### Common Challenges & Limitations

Even with sophisticated models like MOG2, background subtraction faces several hard real-world problems:

**Illumination Changes** — A sudden light switch or cloud passing over can cause the entire frame's pixel intensities to shift simultaneously. The background model sees this as "everything became foreground" before it adapts.

**Dynamic Backgrounds** — Waving trees, water reflections, and flickering displays are correctly identified as "changing" by the model — which is statistically accurate — but they're not the objects we want to detect.

**Ghost Objects** — When a foreground object stops moving, it begins to look like the new background. MOG2 will gradually incorporate it into the background model. When the object eventually moves again, there's a brief "ghost" effect as the old pixels are still in the model.

**Shadows** — Cast shadows move with an object and change pixel intensities, but they're not the object. Without shadow detection, shadows cause significant false positives and make bounding boxes artificially large.

**Camouflage** — If an object has very similar color or texture to the background, the pixel-level difference is small and the subtractor won't detect it.

**Camera Motion** — Background subtraction fundamentally assumes the camera is static. A moving camera makes the *entire background* look like foreground. Camera stabilization must be applied first.

Understanding these limitations is important — they define when to use background subtraction and when to reach for something else.

---

## Part 3 — Motion Detection

### What is Motion Detection?

Motion detection is the next layer up from background subtraction. Where background subtraction asks "which pixels changed?", motion detection asks "where are the moving objects, and what can we say about them?"

There's a useful hierarchy of three levels:

**Level 1 — Pixel Level.** This is what background subtraction gives us. A binary mask showing which pixels are foreground. The question answered here is: *which pixels changed?*

**Level 2 — Region Level.** Group the foreground pixels into coherent regions, find their outlines, and draw bounding boxes around them. The question answered here is: *where are the objects and how big are they?*

**Level 3 — Semantic Level.** Classify the detected objects, track them across frames, infer their behavior. The question answered here is: *what are these objects and what are they doing?*

This session focuses on Levels 1 and 2 — the classical computer vision pipeline. Level 3 typically involves deep learning and will be covered in a future session.

---

### The Classical Motion Detection Pipeline

Here's the full pipeline from raw video frame to detected bounding boxes:

**Step 1 — Capture Frame.** Read a frame from the video using `cv2.VideoCapture`.

**Step 2 — Background Subtraction.** Apply MOG2 or KNN to get the foreground mask `fgMask`. At this point you have a binary image — white where motion is detected, black everywhere else. But it's noisy.

**Step 3 — Morphological Cleanup.** The raw foreground mask has noise: small isolated white pixels from sensor noise, holes inside larger blobs where the object happened to be similar to the background. We clean this up with morphological operations:
- **Erosion** removes small isolated blobs (noise)
- **Dilation** fills in holes and reconnects fragmented regions

Applied in sequence (erode then dilate = *opening*), this gives a much cleaner mask to work with.

**Step 4 — Contour Detection.** Call `cv2.findContours()` on the cleaned mask. This returns a list of contours — curves tracing the boundary of each white blob. Using `cv2.RETR_EXTERNAL` ensures we only get the outermost boundary of each region, avoiding nested duplicate contours.

**Step 5 — Filter by Area.** Many small contours will still remain — specks, edge artifacts, tiny movements. Filter these out by checking `cv2.contourArea(cnt)` against a minimum threshold (e.g., 500 pixels). Anything smaller is discarded.

**Step 6 — Bounding Box.** For each remaining contour, call `cv2.boundingRect()` to get the axis-aligned bounding rectangle — `(x, y, width, height)`. Draw this on the original frame with `cv2.rectangle()`.

This six-step pipeline is the backbone of most classical motion detection systems.

---

### Contours and Bounding Boxes

Let's look at the two key operations more carefully.

**Contours** are curves that join all continuous points along a boundary that share the same color or intensity. In our context, a contour is the outline of a white blob in the foreground mask.

```python
contours, _ = cv2.findContours(
    fgMask, cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE)
```

`cv2.CHAIN_APPROX_SIMPLE` compresses horizontal, vertical, and diagonal segments to just their endpoints, saving memory. `cv2.RETR_EXTERNAL` returns only outermost contours — avoids nested duplicates when one blob contains holes.

**Bounding Rectangles** are the smallest axis-aligned rectangles that completely enclose a contour. They give you the location `(x, y)` of the top-left corner, and the `width` and `height` of the detected object.

```python
x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

The standard pattern is to filter first, then draw:

```python
for cnt in contours:
    if cv2.contourArea(cnt) < 500:
        continue
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

This is essentially the entire detection loop for a simple motion detector.

---

### Where These Techniques Are Used

Let's ground all of this in real applications — these three techniques are everywhere in deployed systems.

**Traffic Monitoring** — Fixed cameras above roads use background subtraction to detect vehicles, then motion detection to count them, measure speeds, and analyze flow patterns. Optical flow can additionally estimate per-vehicle velocity vectors.

**Security & Surveillance** — Perimeter cameras use background subtraction to trigger alerts when anything moves in a designated zone. More sophisticated systems detect loitering (a person stays in one region for too long) using region-level tracking.

**Industrial Inspection** — Cameras watching assembly lines use optical flow and motion detection to verify that components are moving correctly and detect anomalies in the production process.

**Sports Analytics** — Player tracking in football, basketball, and tennis uses optical flow for smooth inter-frame tracking and background subtraction to separate players from the pitch or court.

**Drone Navigation** — Drones use optical flow from a downward-facing camera for visual odometry — estimating position and velocity without GPS, particularly useful indoors where GPS is unavailable.

**Healthcare / Elder Care** — Fall detection systems use motion detection to alert caregivers when a person's movement patterns suggest they've fallen. Daily activity monitoring uses the same pipeline to understand behavioral patterns over time.

---

## Connecting the Three Techniques

These three techniques are not independent tools — they form a natural hierarchy and are frequently combined:

**Optical Flow** is the most general: it works with any camera motion, computes per-pixel velocity vectors, and requires no background model. It's computationally flexible (sparse or dense) and camera-motion aware.

**Background Subtraction** is specialized for static cameras: it builds a statistical model of the background and produces a binary foreground mask very efficiently. MOG2 and KNN are robust, well-tested implementations. It handles the segmentation problem extremely well when the camera isn't moving.

**Motion Detection** builds on background subtraction: it takes the binary mask as input and produces the region-level outputs that are actually useful — bounding boxes, object counts, event triggers. It's the bridge between raw pixel changes and actionable information.

In a typical deployed system: background subtraction runs every frame to produce the mask; morphological cleanup refines it; contour detection and bounding box drawing produce the detections; and optical flow may be used in parallel for velocity estimation or to improve tracking between detections.

---

## Session Summary

**Optical Flow:**
We started with the brightness constancy constraint and derived the optical flow equation — one equation with two unknowns, leading to the aperture problem. Lucas-Kanade resolves this with a patch-based least squares approach, giving sparse flow at feature points. Farneback resolves it through polynomial expansion to produce a dense flow field over all pixels, visualized using HSV color mapping.

**Background Subtraction:**
The core idea is building a background model and flagging deviations as foreground. We went from simple frame differencing (fragile) to Running Average (adaptive but single-mode) to Mixture of Gaussians (multi-modal, robust). MOG2 is the practical standard in OpenCV — adaptive K, shadow detection, and online learning. Key challenges are lighting changes, dynamic backgrounds, shadows, and camera motion.

**Motion Detection:**
A three-level hierarchy: pixels → regions → semantics. The classical pipeline is: background subtraction → morphological cleanup → contour detection → area filtering → bounding box. This six-step pipeline is the foundation of most surveillance and counting systems.

**Next Session:** Video Analytics II — Object Tracking, DeepSORT, and Action Recognition. We'll extend what we built today into the semantic level: assigning IDs to detected objects and following them across time.

---

## References & Further Reading

### Optical Flow
- [OpenCV Optical Flow Tutorial](https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
- Lucas, B.D. & Kanade, T. (1981). *An Iterative Image Registration Technique with an Application to Stereo Vision.* IJCAI.
- Farneback, G. (2003). *Two-Frame Motion Estimation Based on Polynomial Expansion.* SCIA.

### Background Subtraction
- [OpenCV Background Subtraction Tutorial](https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html)
- Stauffer, C. & Grimson, W.E.L. (1999). *Adaptive Background Mixture Models for Real-Time Tracking.* CVPR.
- [Wikipedia — Background Subtraction](https://en.wikipedia.org/wiki/Background_subtraction)

### Motion Detection
- Piccardi, M. (2004). *Background Subtraction Techniques: A Review.* IEEE International Conference on Systems, Man and Cybernetics.
- [Wikipedia — Optical Flow](https://en.wikipedia.org/wiki/Optical_flow)
