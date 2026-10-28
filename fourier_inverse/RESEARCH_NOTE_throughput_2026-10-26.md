# Research Note — Rapid-Acquisition, Offline-Reconstruction Ultrasound Tomography
**Project:** Blind Full-Waveform Inversion (FWI) for Quantitative Ultrasound Tomography  
**Date:** 26 October 2026  
**Status:** Engineering concept supported by simulation; physical acquisition and clinical throughput remain unvalidated.

> Recorded at the user's request as the project's stated aim: **patient throughput** (short
> patient-facing acquisition, offline reconstruction) is the intended strength of this approach.
> Note text below is as supplied by the user; annotations against the logged evidence are at the end.

## 1. Central Concept

The proposed approach separates **patient-facing acoustic acquisition** from **computational image reconstruction**.

Unlike an imaging workflow that requires reconstruction to finish before the scanner is released, ultrasound tomography can record the complete acoustic dataset, release the patient, and perform computationally intensive FWI afterward.

The primary optimization objective therefore shifts from minimizing reconstruction latency alone to:

- Minimizing patient examination and scanner occupancy time.
- Maximizing scanner throughput without compromising measurement quality.
- Performing quantitative reconstruction asynchronously on dedicated computing resources.
- Maintaining reconstruction and reporting capacity sufficient to prevent downstream bottlenecks.

**Central hypothesis:** A rapid-acquisition, offline-reconstruction architecture could improve patient comfort and scanner throughput for acoustically accessible soft tissues without requiring real-time FWI.

## 2. Proposed Acquisition Architecture

Current numerical configuration:

| Parameter | Configuration |
|---|---|
| Transmitters | 32 |
| Receivers | 325–512, recorded in parallel |
| Reconstruction frequencies | 0.15, 0.30, 0.50, 0.80 MHz |
| Excitation | Proposed broadband ultrasound pulse |
| Source acquisition | Sequential transmitters |
| Reconstruction | Multiresolution frequency-domain FWI |
| Current reconstruction | Static 2D slice |
| Processing | Offline GPU computation |

A single broadband excitation may contain all four frequency components. Each receiver records the time-domain waveform, from which complex frequency-domain measurements can be extracted using Fourier analysis.

Consequently, **four reconstruction frequencies do not necessarily require four acquisition passes**.

With 32 sequential broadband excitations:

- 325 receivers yield **41,600 complex frequency-domain measurements**.
- 512 receivers yield **65,536 complex frequency-domain measurements**.

These are measurement samples, not necessarily independent information dimensions.

All transmitters firing simultaneously would require additional source encoding and decoding. Given the current negative simultaneous-source encoding results, sequential broadband transmission is the preferred initial architecture.

## 3. Acquisition Time Versus Reconstruction Time

The short acoustic propagation time permits rapid data collection in principle.

| Parameter | ~15 cm ring | ~37 cm ring |
|---|---:|---:|
| One-way acoustic crossing | ~0.10 ms | ~0.25 ms |
| Assumed firing interval (3 crossings) | ~0.30 ms | ~0.75 ms |
| 32 sequential firings | ~10 ms | ~24 ms |
| Ten repeated acquisitions | ~0.10 s | ~0.24 s |

These are **idealized lower-bound estimates**, not measured device performance.

Real acquisition time will depend on transducer bandwidth, reverberation decay, electronics, switching, averaging requirements, acoustic coupling and motion.

The current corrected numerical reconstruction takes approximately **three minutes per 2D slice on a Tesla T4** under its tested conditions. This processing time does not require continued patient immobilization.

However, full-volume reconstruction could create substantial computational demand. For example, 50 slices at three minutes per slice would require approximately 150 minutes of serial GPU processing unless computation is accelerated or parallelized.

## 4. Potential Clinical Throughput Advantage

The proposed workflow is:

**Patient positioning → acoustic acquisition → measurement quality check → patient release → offline reconstruction → image interpretation.**

This allows another patient to be examined while the preceding patient's acoustic dataset is reconstructed.

The potential benefits include reduced motion-sensitive examination time, improved patient comfort, reduced scanner occupancy, and increased examination capacity.

However, scanner throughput and report turnaround are separate outcomes. A short scan may still generate a long reconstruction or reporting queue.

The operational requirements are therefore:

1. Acquisition and patient turnover must be sufficiently short.
2. Data quality must be assessed before releasing the patient to minimize recalls.
3. Reconstruction capacity must keep pace with incoming examinations.
4. Image interpretation and reporting must not become the dominant bottleneck.

The approach should be evaluated through **patients examined per scanner-hour, time to diagnostic report, repeat-scan rate, and cost per completed examination**.

## 5. Anatomical Applications

The feasibility of rapid acquisition depends on accessible acoustic paths rather than simply the absence of hard tissue.

**Most favorable:**
- Breast tomography.
- Superficial soft-tissue structures.
- Extremity muscle and soft-tissue imaging.

**Potentially feasible but more challenging:**
- Selected solid abdominal organs.
- Pelvic and gynecological structures.
- Obstetric applications, subject to motion, acoustic access and safety validation.

**Particularly difficult:**
- Transthoracic cardiac tomography due to aerated lungs, ribs and cardiac motion.
- Imaging through extensive bone or gas obstruction.

For the first physical prototype, breast or extremity imaging would provide a more favorable validation environment than transthoracic cardiac imaging.

## 6. Current Evidence and Limitations

The numerical FWI pipeline has demonstrated reconstruction of patient-anatomy-derived synthetic phantoms without supplying their true anatomical labels or sound-speed map to the inversion.

In the corrected cardiac simulation, the external heart contour achieved a Dice coefficient of approximately **0.985**, while thick myocardial interiors achieved a sound-speed RMSE of approximately **2.3 m/s**.

However, blood-pool and thin-layer segmentation remain less accurate, and these results come from one static 2D water-bath setting with synthetic acoustic measurements.

These findings support further development, but **do not establish clinical equivalence to CT, MRI or existing ultrasound tomography systems**.

Likewise, rapid waveform propagation does not establish that the complete examination will be faster than existing clinical scanners. Patient positioning, coupling, volumetric coverage and image quality must be included in future comparisons.

## 7. Required Validation

The next engineering milestones are:

1. **Broadband acquisition:** Verify sufficient acoustic output and receiver sensitivity at all four reconstruction frequencies using one excitation.
2. **Minimum firing interval:** Measure waveform decay and source-to-source contamination to establish the shortest reliable acquisition cycle.
3. **Physical phantom acquisition:** Reconstruct data from real transducers rather than exclusively simulated waveforms.
4. **Motion tolerance:** Determine how acquisition duration and displacement affect reconstruction accuracy.
5. **Volumetric imaging:** Quantify out-of-plane propagation, elevational resolution and scan-step requirements.
6. **Throughput benchmarking:** Measure patient setup time, acquisition time, reconstruction queue length, reporting delay and operational cost.
7. **Diagnostic benchmarking:** Compare anatomical and quantitative accuracy against existing clinical modalities for a predefined clinical task.

## 8. Research Positioning

The core acoustic tomography and full-waveform inversion methods are already established. Rapid acquisition followed by offline reconstruction is also not inherently novel.

A potential contribution would therefore need to demonstrate a measurable operational or technical advantage—for example:

- Comparable anatomical accuracy with fewer independent transmissions.
- Reduced scanner occupancy under clinically realistic conditions.
- Improved acquisition efficiency without excessive reconstruction cost.
- More robust quantitative imaging for a specific soft-tissue application.

Such advantages must be demonstrated against existing ultrasound tomography approaches under comparable conditions.

## Conclusion

**The proposed direction is an acquisition-centered clinical workflow rather than a real-time reconstruction system.**

The fundamental premise is that collecting ultrasound waveforms can be substantially faster than reconstructing the corresponding quantitative image. By separating those stages, a patient could complete the examination and leave while reconstruction continues independently.

The strongest prospective benefit is not necessarily superior spatial resolution to CT or MRI, but **short patient-facing examinations, potentially greater scanner throughput, and quantitative soft-tissue imaging without ionizing radiation**.

This is a plausible product-development direction, but its clinical and operational advantages remain hypotheses until demonstrated using physical acquisition hardware, volumetric imaging and matched clinical benchmarks.

---

## Annotations against the logged evidence (Claude, 2026-10-26)

- **Throughput vs MRI vs CT.** The advantage is strongest against MRI, where exams take tens
  of minutes with long motion-sensitive acquisitions. Against CT it is NOT speed: CT acquires a
  volume in seconds. The advantages over CT are no ionizing radiation and quantitative acoustic
  tissue contrast. Frame the claim as "MRI-like quantitative soft-tissue imaging at CT-like (or
  faster) patient-facing time, without radiation", and do not claim faster than CT.
- **Receiver counts.** Tested configurations were 128, 256 and 512 receivers (LOG run -27). 325
  was not tested; it is a nominal value near the ~2ka ≈ 300 angular-sampling estimate. The
  measurement counts (32 × 325 × 4 = 41,600; 32 × 512 × 4 = 65,536) are arithmetically
  correct.
- **One broadband shot for all frequencies** matches the simulation: every time-domain dataset in
  runs -19…-35 used one broadband pulse per source, with the four frequencies extracted by DFT.
- **Reconstruction time.** 145 s (patient023 slice, 260² final grid, run -25) and 190 s (MRI
  heart, 340² final grid, run -35) on a T4, after the source-scaling / PML fixes (runs -33/-34).
  Algorithm-level speedups are exhausted (runs -25…-28). Remaining levers are hardware (A100/H100
  ~3–4×), amortizing compilation across a slice stack (~14 s/slice), and parallel slices across
  GPUs, which turns the 150 min/50-slice example into minutes on a small cluster.
- **Repeated acquisitions.** Averaging helps only up to ~40 dB in the water bath (run -25), so
  ~10 repeats is about the useful maximum there. Through attenuating paths more may be needed.
- **Motion tolerance (not yet tested).** Coherence needs displacement ≲ λ/10–λ/20 during the
  slice: ~0.1–0.2 mm at 0.8 MHz. A resting limb or a pendant breast should meet this over
  10–100 ms. Abdomen and pelvis need breath-hold or gating; obstetrics has fetal motion.
- **Metrics quoted in §6** (Dice 0.985, thick-myocardium RMSE 2.3 m/s) are from run -36:
  single static 2D slice, water bath, 30 dB, synthetic data from an independent time-domain
  simulator, threshold segmentation. Blood-pool Dice 0.842 against a segmentation ceiling of
  0.853.
- **Gas** (lung, bowel) cannot be bypassed by frequency (run -35 analysis). Abdominal and
  obstetric feasibility hinges on gas-free acoustic paths.
