This document outlines the current adaptive glyph detection algorithm, which aims to produce a binary map where white pixels represent glyphs and black pixels represent the background. The strategy uses polynomial approximation error to guide a quadtree-based segmentation and classification process.

### **Algorithm Steps:**

1. **Initial Global Pass (AdaptiveEdgeDetector.run\_detection):**  
   * **a. Load Image:** The original\_image (grayscale, normalized to \[0,1\]) is loaded.  
   * **b. Global Simple Detector Initialization:** A SimpleEdgeDetector instance (self.global\_simple\_detector) is created and configured with the global image parameters.  
   * **c. Global Image Processing:** self.global\_simple\_detector.process\_full\_image() is called. This performs polynomial approximation on the *entire* image as a single segment. It calculates various raw error maps (e.g., error\_original, error\_smoothed), combines them based on ERROR\_COMBINATION\_STRATEGY (e.g., 'logical\_and'), and applies a global threshold (EDGE\_THRESHOLD\_TYPE, FIXED\_EDGE\_THRESHOLD) to produce a final\_binary\_edge\_map for the whole image. It also stores the raw\_error\_maps for the full image.  
   * **d. Global Error Check:** The global\_raw\_error\_map (specifically 'error\_original') is retrieved from self.global\_simple\_detector, and its global\_error\_measure (MSE or MAE) is calculated.  
   * **e. Initial Map Decision:**  
     * **If global\_error\_measure \< self.config.GLOBAL\_ERROR\_THRESHOLD:** The entire image is considered too smooth to contain significant glyphs. self.final\_adaptive\_glyph\_map is initialized as an all-black image (np.zeros\_like). The adaptive process terminates here.  
     * **Else (global\_error\_measure \>= self.config.GLOBAL\_ERROR\_THRESHOLD):** The image is sufficiently complex to warrant adaptive processing. self.final\_adaptive\_glyph\_map is initialized as an all-black image (np.zeros\_like). The recursive segmentation process begins.  
2. Recursive Segment Processing (\_process\_segment\_recursively):  
   This function is called for each segment, starting with the entire image, and recursively for sub-segments.  
   * **a. Local Simple Detector Initialization and Processing:**  
     * For the current segment\_image\_data, a *new* SimpleEdgeDetector instance (segment\_detector) is created.  
     * This segment\_detector then processes *only* the current segment\_image\_data (as if it were a full image). This calculates local raw error maps and a local error measure for this specific segment.  
     * The local\_raw\_error\_map (specifically 'error\_original') is retrieved, and its local\_error\_measure is calculated.  
   * **b. Decision Logic (Order of Conditions is Critical):**  
     * **Condition 1: Smooth Region \- Natural Termination (Low Local Error):**  
       * **Check:** if local\_error\_measure \< self.config.LOCAL\_ERROR\_THRESHOLD:  
       * **Action:** This segment is considered locally smooth. A decision is made to classify it as either glyph interior or background based on its mean pixel intensity.  
         * Calculate mean\_intensity \= np.mean(segment\_image\_data).  
         * **If mean\_intensity \> self.config.GLYPH\_INTENSITY\_THRESHOLD:**  
           * The segment is classified as a **glyph interior**. The corresponding area in self.final\_adaptive\_glyph\_map is set to **white (1)**.  
           * Its status is recorded as 'terminated\_low\_error\_glyph'.  
         * **Else (mean\_intensity \<= self.config.GLYPH\_INTENSITY\_THRESHOLD):**  
           * The segment is classified as **background**. The corresponding area in self.final\_adaptive\_glyph\_map is set to **black (0)**.  
           * Its status is recorded as 'terminated\_low\_error\_background'.  
         * The recursion for this branch terminates.  
     * **Condition 2: Forced Termination (Minimum Size or Maximum Depth):**  
       * **Check:** if height \<= self.config.MIN\_SEGMENT\_SIZE or width \<= self.config.MIN\_SEGMENT\_SIZE or current\_depth \>= self.config.MAX\_DEPTH:  
       * **Action:** This segment has reached a predefined limit for subdivision. A final decision is made for it based on its mean pixel intensity, similar to the low-error termination.  
         * Calculate mean\_intensity \= np.mean(segment\_image\_data).  
         * **If mean\_intensity \> self.config.GLYPH\_INTENSITY\_THRESHOLD:**  
           * The segment is classified as a **glyph interior**. The corresponding area in self.final\_adaptive\_glyph\_map is set to **white (1)**.  
           * Its status is recorded as 'terminated\_by\_size\_or\_depth\_glyph'.  
         * **Else (mean\_intensity \<= self.config.GLYPH\_INTENSITY\_THRESHOLD):**  
           * The segment is classified as **background**. The corresponding area in self.final\_adaptive\_glyph\_map is set to **black (0)**.  
           * Its status is recorded as 'terminated\_by\_size\_or\_depth\_background'.  
         * The recursion for this branch terminates.  
     * **Condition 3: High Error \- Subdivision (Default Case if not Terminated):**  
       * **Check:** (This condition is implicitly met if neither Condition 1 nor Condition 2 is true, meaning local\_error\_measure \>= self.config.LOCAL\_ERROR\_THRESHOLD and the segment can still be subdivided).  
       * **Action:** This segment is considered complex and requires finer resolution.  
         * The self.final\_adaptive\_glyph\_map is **NOT updated** at this stage. The idea is that this segment will be refined by its children, and their termination decisions will populate the map.  
         * Its status is recorded as 'subdividing'.  
         * The current segment is divided into four equal sub-segments (quadtree).  
         * \_process\_segment\_recursively is called for each of these four sub-segments, incrementing current\_depth.  
3. **Result Visualization and Saving (AdaptiveEdgeDetector.save\_results, AdaptiveEdgeDetector.plot\_results):**  
   * After the recursive process completes, the final\_adaptive\_glyph\_map contains the accumulated glyph/background classifications.  
   * This map, along with the original image and a heatmap showing the status of each terminal segment, is saved and plotted for analysis.

This algorithm emphasizes classifying regions as glyphs or background only at the point of termination (either due to smoothness or reaching a resolution limit), and avoids introducing "edge" information into the final glyph map during subdivision. The GLYPH\_INTENSITY\_THRESHOLD is crucial for distinguishing between light and dark regions.