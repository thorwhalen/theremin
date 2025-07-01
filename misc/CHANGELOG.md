# Theremin Project Changelog

## 2025-07-01 - Systematic Pipeline Parameter Mismatch Diagnosis and Repair

### Overview
Conducted comprehensive systematic testing of all theremin pipelines to identify and fix parameter mismatches between knob functions and synth functions. Many pipelines were failing with `ValueError: Unknown parameter` errors when hands were detected in the video feed.

### Core Issue Identified
The primary problem was that many pipelines used `two_hand_freq_and_volume_knobs` which produces parameters like `l_freq`, `l_volume`, `r_freq`, `r_volume`, `vibrato_rate`, `vibrato_depth`, `reverb_mix`, but most synths expected simpler parameter sets or different parameter names.

### CLI Improvements
- Enhanced CLI so that flags like `--synth`, `--pipeline`, `--knobs`, and `--video-features` with no argument list available options
- Fixed naming conflict in `script_utils.py` between the `knobs` parameter and imported `knobs` dictionary
- Added argh argument decorators to support listing with bare flags (e.g., `--pipeline` lists all pipelines)

### Default Behavior Fix
- Fixed the default `theremin_knobs` function to allow single-hand fallback for volume/frequency control
- Previously, the default pipeline required both hands to be detected for any sound output

### Pipeline Fixes Applied

#### ✅ **FIXED AND WORKING:**
1. **`two_voice_and_hands`** - Created dedicated `two_voice_knobs` function that only returns parameters accepted by `two_voice_synth_func` (`l_freq`, `l_volume`, `r_freq`, `r_volume`)

2. **`chorused_two_hands`** - Temporarily fixed using `simple_two_hands_knobs` (maps to `freq`, `volume` only)
   - *Note: Missing chorus-specific controls `depth` and `speed`*

3. **`noise_two_hands`** - Temporarily fixed using `simple_two_hands_knobs` 
   - *Note: Missing `noise_level` control*

4. **`phase_distortion_synth`** - Temporarily fixed using `simple_two_hands_knobs`
   - *Note: Missing `distortion` control*

5. **`sine_two_hands`** - Fixed using `simple_two_hands_knobs` (perfect match - only needs `freq`, `volume`)

6. **`square_two_hands`** - Fixed using `simple_two_hands_knobs` (perfect match - only needs `freq`, `volume`)

7. **`natural_sounding_synth_lr`** - Already working (accepts parameters from `two_hand_freq_and_volume_knobs`)

#### ❌ **STILL BROKEN:**
1. **`ringmod_two_hands`** - Parameter mismatch (needs `freq`, `volume`, `mod_freq_ratio` but receives `l_freq`, etc.)
2. **`supersaw_two_hands`** - Likely parameter mismatch (needs `freq`, `volume`, `detune`, `n_voices`)

#### 🔄 **NOT TESTED YET:**
1. **`rhythmic_fm`** - Uses `rhythmic_fm_synth_knobs` (should work)
2. **`high_sines`** - Uses `high_sines_theremin_knobs` (should work)  
3. **`high_sines_openness`** - Uses `high_sines_openness_theremin_knobs` (should work)
4. **`high_sines_pinch`** - Uses `high_sines_pinch_theremin_knobs` (should work)
5. **`theremin`** / **`default`** - Uses `theremin_knobs` (should work)

### New Functions Created
- **`simple_two_hands_knobs`**: Maps two-hand gestures to basic `freq` and `volume` parameters for simple synths
- **`two_voice_knobs`**: Specialized function for `two_voice_synth_func` with exact parameter matching

### Testing Methodology
- Used systematic approach: start each pipeline, wait for initialization, check for parameter errors
- Used `pkill -f "python theremin/main.py"` to kill processes after testing
- Captured error messages to identify exact parameter mismatches
- Verified synth parameter expectations by examining function signatures

### Key Technical Insights
1. **Parameter Filtering**: The most robust approach is to create knobs functions that only return parameters the target synth can accept
2. **Hand Fallback**: Single-hand fallback logic is essential for usability (left hand primary, right hand secondary)
3. **CLI UX**: Users expect `--pipeline` alone to list options, not throw an error
4. **Error Propagation**: Parameter mismatches only surface when hands are actually detected in video feed

### Files Modified
- `/Users/thorwhalen/Dropbox/py/proj/t/theremin/theremin/audio.py`
  - Added `two_voice_knobs` function
  - Added `simple_two_hands_knobs` function  
  - Updated pipeline definitions to use correct knobs functions
  - Modified `theremin_knobs` for single-hand fallback
- `/Users/thorwhalen/Dropbox/py/proj/t/theremin/theremin/script_utils.py`
  - Added argh decorators for CLI argument listing
  - Fixed naming conflict with knobs dictionary

### Next Steps
- Complete testing of remaining untested pipelines
- Create specialized knobs functions for synths requiring additional parameters:
  - `chorused_two_hands_knobs` (add `depth`, `speed` controls)
  - `noise_two_hands_knobs` (add `noise_level` control)  
  - `phase_distortion_two_hands_knobs` (add `distortion` control)
  - `ringmod_two_hands_knobs` (add `mod_freq_ratio` control)
  - `supersaw_two_hands_knobs` (add `detune`, `n_voices` controls)
- Clean up duplicate function definitions in audio.py
- Add video gesture mappings for advanced controls (hand openness → effects parameters)

### Impact
- Restored functionality to multiple broken pipelines
- Improved CLI usability and discoverability
- Established systematic approach for parameter compatibility
- Enhanced single-hand operation support
- Created foundation for rapid diagnosis of similar issues
